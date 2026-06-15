import argparse
from itertools import combinations
from pathlib import Path

import polars as pl
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils.distance import (
    calc_batched_bures_wasserstein_squared_distance,
    calc_batched_euclidean_squared_distance,
)
from utils.mylogging import init_logger
from utils.normalize import normalize_for_unit_mean
from utils.pooling import covariance_pooling, mean_pooling
from utils.seed import fix_seeds
from utils.socm import calc_socm

logger = init_logger(__name__, log_file_path=Path("logs/calc_socm_pairwise.log"))
fix_seeds(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=Path, required=True, help="Input CSV file path."
    )
    parser.add_argument(
        "--output_file", type=Path, required=True, help="Output CSV file path."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Model name.",
    )
    parser.add_argument("--cuda_id", type=int, default=0, help="CUDA device ID.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    return parser.parse_args()


def encode_embeddings(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode texts into mean and covariance of normalized token embeddings.

    Parameters
    ----------
    model : AutoModel
    tokenizer : AutoTokenizer
    texts : list[str]
    batch_size : int
    device : torch.device

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        all_mus: (N, D), all_sigmas: (N, D, D).
    """
    all_mus, all_sigmas = [], []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        inputs = tokenizer(
            texts[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.to(torch.float32)
        attention_mask = inputs["attention_mask"]
        normalized = normalize_for_unit_mean(last_hidden_states, attention_mask)
        all_mus.append(mean_pooling(normalized, attention_mask))
        all_sigmas.append(covariance_pooling(normalized, attention_mask))
    return torch.cat(all_mus, dim=0), torch.cat(all_sigmas, dim=0)


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
    text_ids = df["text_id"].to_list()
    texts = df["text"].to_list()
    logger.info(f"Number of texts: {len(texts)}")

    logger.info("Building model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(
        device
    )

    logger.info("Encoding texts...")
    all_mus, all_sigmas = encode_embeddings(
        model, tokenizer, texts, args.batch_size, device
    )
    all_mus = all_mus.to(torch.float32)
    all_sigmas = all_sigmas.to(torch.float32)

    pair_ids = list(combinations(range(len(texts)), 2))
    num_pairs = len(pair_ids)
    idx_i = torch.tensor([i for i, _ in pair_ids], device=device)
    idx_j = torch.tensor([j for _, j in pair_ids], device=device)

    logger.info("Calculating d_mu...")
    d_mu_list = []
    for start in tqdm(range(0, num_pairs, args.batch_size)):
        end = min(start + args.batch_size, num_pairs)
        euclidean_sq = calc_batched_euclidean_squared_distance(
            all_mus[idx_i[start:end]], all_mus[idx_j[start:end]]
        )
        d_mu_list.append(euclidean_sq / 4)
    d_mu_vals = torch.cat(d_mu_list, dim=0)

    logger.info("Calculating d_sigma...")
    d_sigma_list = []
    for start in tqdm(range(0, num_pairs, args.batch_size)):
        end = min(start + args.batch_size, num_pairs)
        bw_sq = calc_batched_bures_wasserstein_squared_distance(
            all_sigmas[idx_i[start:end]], all_sigmas[idx_j[start:end]]
        )
        d_sigma_list.append(bw_sq / 4)
    d_sigma_vals = torch.cat(d_sigma_list, dim=0)

    logger.info("Calculating SOCM...")
    socm_vals = calc_socm(d_mu_vals.cpu(), d_sigma_vals.cpu())

    results = pl.DataFrame(
        {
            "text1_id": [text_ids[i] for i, _ in pair_ids],
            "text2_id": [text_ids[j] for _, j in pair_ids],
            "d_mu": d_mu_vals.cpu().tolist(),
            "d_sigma": d_sigma_vals.cpu().tolist(),
            "socm": socm_vals.tolist(),
        }
    )
    results.write_csv(args.output_file)
    logger.info(f"Saved results to {args.output_file}")

    avg_socm = socm_vals.mean().item()
    print(f"Avg. SOCM: {avg_socm:.3f}")


if __name__ == "__main__":
    main()
