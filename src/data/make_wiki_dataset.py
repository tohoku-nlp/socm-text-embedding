import argparse
import random
import urllib.request
from pathlib import Path

import polars as pl

from utils.seed import fix_seeds
from utils.mylogging import init_logger

logger = init_logger(__name__, log_file_path=Path("logs/make_wiki_dataset.log"))
fix_seeds(0)

WIKI_URL = "https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Wikipedia dataset and sample random lines."
    )
    parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Path to save the sampled_lines output CSV file.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1000,
        help="Number of lines to sample (default: 1000).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # download
    logger.info(f"Downloading from {WIKI_URL} ...")
    with urllib.request.urlopen(WIKI_URL) as response:
        lines = [line.decode("utf-8").strip() for line in response]

    # random sample
    sampled_lines = random.sample(lines, args.num_samples)

    # save as CSV
    df = pl.DataFrame({"text_id": range(len(sampled_lines)), "text": sampled_lines})
    df.write_csv(args.output_file)
    logger.info(f"Saved {len(sampled_lines)} lines -> {args.output_file}")


if __name__ == "__main__":
    main()
