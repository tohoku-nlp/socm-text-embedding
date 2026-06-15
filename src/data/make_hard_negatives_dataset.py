import argparse
from pathlib import Path

import polars as pl
from datasets import load_dataset

from utils.seed import fix_seeds
from utils.mylogging import init_logger

logger = init_logger(
    __name__, log_file_path=Path("logs/make_hard_negatives_dataset.log")
)
fix_seeds(0)

DATASET_NAME = "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MS MARCO hard negatives and sample to CSV."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to save the sampled output CSV file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of samples (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # load dataset
    logger.info("Loading MS MARCO hard negatives ...")
    ds = load_dataset(DATASET_NAME, "triplet-hard", split="train")
    logger.info(f"Loaded {len(ds)} samples.")

    # random sample
    num_samples = min(args.num_samples, len(ds))
    sampled_ds = ds.shuffle(seed=0).select(range(num_samples))

    # save as CSV
    df = pl.DataFrame(
        {
            "query": sampled_ds["query"],
            "negative": sampled_ds["negative"],
        }
    )
    df.write_csv(args.output_file)
    logger.info(f"Saved {len(df)} samples -> {args.output_file}")


if __name__ == "__main__":
    main()
