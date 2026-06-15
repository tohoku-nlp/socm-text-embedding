import argparse
from pathlib import Path

import mteb
import polars as pl
from sentence_transformers import SentenceTransformer

from utils.mylogging import init_logger
from utils.seed import fix_seeds

logger = init_logger(__name__, log_file_path=Path("logs/run_mteb.log"))
fix_seeds(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MTEB benchmark with a specified model."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="h-tomo/unsup-simcse-bert-base-uncased-mean",
        help="Name of the pretrained transformer model.",
    )
    parser.add_argument(
        "--benchmark_name",
        type=str,
        default="MTEB(eng, v2)",
        help="Name of the MTEB benchmark to evaluate on.",
    )
    parser.add_argument(
        "--output_file", type=Path, required=True, help="Output CSV file path."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for encoding."
    )
    parser.add_argument(
        "--cuda_id",
        type=int,
        default=0,
        help="CUDA device ID to use for computation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info(f"Model: {args.model_name}")
    logger.info(f"Benchmark: {args.benchmark_name}")

    logger.info(f"Building model: {args.model_name}...")
    model = SentenceTransformer(
        model_name_or_path=args.model_name,
        prompts={"query": "", "passage": ""},
        default_prompt_name=None,
        device=f"cuda:{args.cuda_id}",
        trust_remote_code=True,
    )
    # Ensure the model uses mean pooling
    assert any(
        getattr(module, "pooling_mode_mean_tokens", False) for module in model.modules()
    ), "pooling_mode_mean_tokens is not enabled"

    logger.info(f"Loading benchmark: {args.benchmark_name}...")
    benchmark = mteb.get_benchmark(args.benchmark_name)

    logger.info("Evaluating model on benchmark...")
    results = mteb.evaluate(
        model,
        tasks=benchmark,
        encode_kwargs={
            "batch_size": args.batch_size,
            "device": f"cuda:{args.cuda_id}",
        },
    )

    logger.info(f"Revision: {results.model_revision}")
    logger.info(f"Num tasks: {len(results.task_results)}")
    assert len(results.task_results) == 41
    score = sum(r.scores["test"][0]["main_score"] for r in results.task_results) / len(
        results.task_results
    )
    df = pl.DataFrame({"mteb_score": [score]})
    df.write_csv(args.output_file)
    logger.info(f"Saved results to {args.output_file}")


if __name__ == "__main__":
    main()
