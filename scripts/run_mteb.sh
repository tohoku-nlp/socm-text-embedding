mkdir -p results/mteb

model_names=(
    "google-bert/bert-base-uncased"
    "h-tomo/unsup-simcse-bert-base-uncased-mean"
    "thenlper/gte-base"
    "intfloat/e5-base-v2"
    "microsoft/MiniLM-L12-H384-uncased"
    "sentence-transformers/all-MiniLM-L12-v2"
    "intfloat/e5-small-v2"
    "thenlper/gte-small"
    "microsoft/mpnet-base"
    "sentence-transformers/all-mpnet-base-v2"
    "nomic-ai/nomic-bert-2048"
    "nomic-ai/nomic-embed-text-v1.5"
)

for model_name in "${model_names[@]}"; do
    uv run python3 src/run_mteb.py \
        --model_name "${model_name}" \
        --benchmark_name "MTEB(eng, v2)" \
        --output_file "results/mteb/mteb_eng_v2_${model_name//\//-}.csv" \
        --cuda_id 0 \
        --batch_size 64
done
