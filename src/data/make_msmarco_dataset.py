import argparse
from pathlib import Path

import polars as pl
from datasets import load_dataset

from utils.seed import fix_seeds
from utils.mylogging import init_logger

logger = init_logger(__name__, log_file_path=Path("logs/make_msmarco_dataset.log"))
fix_seeds(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download MSMARCO corpus and sample to CSV."
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

    # load corpus
    logger.info("Loading MSMARCO corpus ...")
    corpus = load_dataset("mteb/msmarco", name="corpus")["corpus"]
    logger.info(f"Loaded {len(corpus)} documents.")

    # random sample
    num_samples = min(args.num_samples, len(corpus))
    sampled = corpus.shuffle(seed=0).select(range(num_samples))

    # save as CSV
    df = pl.DataFrame({"text_id": range(num_samples), "text": sampled["text"]})
    df.write_csv(args.output_file)
    logger.info(f"Saved {num_samples} lines -> {args.output_file}")


if __name__ == "__main__":
    main()
