import argparse
from pathlib import Path

import polars as pl
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.mylogging import init_logger
from utils.seed import fix_seeds

logger = init_logger(
    __name__, log_file_path=Path("logs/calc_last_token_concentration.log")
)
fix_seeds(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()

    device = torch.device(
        f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"
    )
    logger.info(f"Dataset: {args.input_file}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Device: {device}")

    df = pl.read_csv(args.input_file)
    texts = df["text"].to_list()
    text_ids = df["text_id"].to_list()
    logger.info(f"Number of texts: {len(texts)}")

    logger.info("Building model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    rows = []
    logger.info(
        "Calculating S(X) / ||mu(X)||^2 "
        "where S(X) = (1/n) sum_j ||x_j - mu(X)||^2..."
    )

    total_batches = (len(texts) + args.batch_size - 1) // args.batch_size

    for start in tqdm(range(0, len(texts), args.batch_size), total=total_batches):
        batch_texts = texts[start : start + args.batch_size]
        batch_text_ids = text_ids[start : start + args.batch_size]

        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        outputs = model(**enc)
        hidden_states = outputs.last_hidden_state  # (B, T, D)

        attention_mask = enc["attention_mask"]  # (B, T)

        for b_idx, text_id in enumerate(batch_text_ids):
            valid_mask = attention_mask[b_idx].bool()
            x = hidden_states[b_idx][valid_mask]  # (n, D)

            seq_len = int(valid_mask.sum().item())
            if seq_len == 0:
                continue

            mu = x.mean(dim=0)  # (D)
            s_x = (x - mu).pow(2).sum(dim=1).mean()
            mu_sq = mu.pow(2).sum()

            sx_over_mu_sq = torch.where(
                mu_sq > 0,
                s_x / mu_sq,
                torch.tensor(0.0, device=device),
            )

            rows.append(
                {
                    "text_id": text_id,
                    "sx_over_mu_sq": float(sx_over_mu_sq.item()),
                }
            )

    out_df = pl.DataFrame(rows)
    out_df.write_csv(args.output_file)
    logger.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
