import argparse
from pathlib import Path

import polars as pl
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.mylogging import init_logger
from utils.seed import fix_seeds

logger = init_logger(__name__, log_file_path=Path("logs/calc_avg_cos.log"))
fix_seeds(0)


def compute_avg_cos(vecs: torch.Tensor, eps: float = 1e-12) -> float:
    """Compute mean pairwise cosine similarity including self-pairs.

    Parameters
    ----------
    vecs : torch.Tensor
        Shape (n, d).
    eps : float
        Minimum norm clamp to avoid division by zero.

    Returns
    -------
    float
        Mean of the full n×n cosine similarity matrix.
    """
    norms = torch.norm(vecs, dim=-1, keepdim=True).clamp_min(eps)
    vecs_unit = vecs / norms
    cos_mat = vecs_unit @ vecs_unit.transpose(0, 1)
    return float(cos_mat.mean().item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--cuda_id", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=64)
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
        output_hidden_states=True,
        trust_remote_code=True,
    )
    model.eval()
    model.to(device)

    rows = []
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
        # hidden_states[0]: embedding output; hidden_states[l+1]: encoder layer l output
        hidden_states_tuple = outputs.hidden_states

        attention_mask = enc["attention_mask"]  # (B, T)

        for hs_idx, hidden_states in enumerate(hidden_states_tuple):
            layer_idx = hs_idx - 1  # -1 for embedding output, 0-indexed thereafter

            for b_idx, text_id in enumerate(batch_text_ids):
                valid_mask = attention_mask[b_idx].bool()
                x = hidden_states[b_idx][valid_mask]  # (n, D)

                if valid_mask.sum() == 0:
                    continue

                rows.append(
                    {
                        "text_id": text_id,
                        "layer": layer_idx,
                        "avg_cos": compute_avg_cos(x),
                    }
                )

    out_df = pl.DataFrame(rows)
    out_df.write_csv(args.output_file)
    logger.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
