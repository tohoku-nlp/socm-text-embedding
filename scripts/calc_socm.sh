mkdir -p results/socm

dataset_names=("wiki_1000" "msmarco_1000")
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

for dataset_name in "${dataset_names[@]}"; do
    for model_name in "${model_names[@]}"; do
        uv run python3 src/calc_socm_pairwise.py \
            --input_file "data/${dataset_name}.csv" \
            --output_file "results/socm/${dataset_name}_${model_name//\//-}.csv" \
            --model_name "${model_name}" \
            --cuda_id 0 \
            --batch_size 64
    done
done

for model_name in "${model_names[@]}"; do
    uv run python3 src/calc_socm_hard_negative.py \
        --input_file "data/hard_negatives_50000.csv" \
        --output_file "results/socm/hard_negatives_50000_${model_name//\//-}.csv" \
        --model_name "${model_name}" \
        --cuda_id 0 \
        --batch_size 64
done
