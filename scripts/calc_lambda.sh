mkdir -p results/lambda

dataset_name="wiki_1000"
model_names=(
    "google-bert/bert-base-uncased"
    "h-tomo/unsup-simcse-bert-base-uncased-mean"
    "thenlper/gte-base"
    "intfloat/e5-base-v2"
    "microsoft/MiniLM-L12-H384-uncased"
    "sentence-transformers/all-MiniLM-L12-v2"
    "intfloat/e5-small-v2"
    "thenlper/gte-small"
)

for model_name in "${model_names[@]}"; do
    uv run python3 src/calc_lambda.py \
        --input_file "data/${dataset_name}.csv" \
        --output_file "results/lambda/${dataset_name}_${model_name//\//-}.csv" \
        --model_name "${model_name}" \
        --cuda_id 0 \
        --batch_size 64
done
