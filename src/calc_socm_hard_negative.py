import argparse
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

logger = init_logger(__name__, log_file_path=Path("logs/calc_socm_hard_negative.log"))
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


def encode_batch(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode a single batch of texts into (mu, sigma) on CPU.

    Parameters
    ----------
    model : AutoModel
    tokenizer : AutoTokenizer
    texts : list[str]
    device : torch.device

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        mus: (B, D), sigmas: (B, D, D) on CPU.
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state.to(torch.float32)
    attention_mask = inputs["attention_mask"]
    normalized = normalize_for_unit_mean(last_hidden_states, attention_mask)
    mu = mean_pooling(normalized, attention_mask).cpu()
    sigma = covariance_pooling(normalized, attention_mask).cpu()
    return mu, sigma


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
    queries = df["query"].cast(pl.String).to_list()
    negatives = df["negative"].cast(pl.String).to_list()
    n = len(queries)
    logger.info(f"Number of pairs: {n}")

    logger.info("Building model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True).to(
        device
    )

    socm_vals: list[float] = []
    for start in tqdm(range(0, n, args.batch_size), desc="Calculating SOCM"):
        end = min(start + args.batch_size, n)

        q_mus, q_sigmas = encode_batch(model, tokenizer, queries[start:end], device)
        neg_mus, neg_sigmas = encode_batch(
            model, tokenizer, negatives[start:end], device
        )

        euclidean_sq = calc_batched_euclidean_squared_distance(
            q_mus.to(device), neg_mus.to(device)
        ).cpu()
        bw_sq = calc_batched_bures_wasserstein_squared_distance(
            q_sigmas.to(device), neg_sigmas.to(device)
        ).cpu()

        d_mu = euclidean_sq / 4
        d_sigma = bw_sq / 4
        for i in range(end - start):
            socm_vals.append(calc_socm(d_mu[i].item(), d_sigma[i].item()))

    results = pl.DataFrame(
        {
            "query": queries,
            "negative": negatives,
            "socm": socm_vals,
        }
    )
    results.write_csv(args.output_file)
    logger.info(f"Saved results to {args.output_file}")

    avg_qn = sum(socm_vals) / n
    print(f"Avg. SOCM: {avg_qn:.3f}")


if __name__ == "__main__":
    main()
