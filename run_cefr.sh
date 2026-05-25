#!/bin/bash
#SBATCH --job-name=cefr_dialect
#SBATCH --account=aip-creager
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --array=0-2

module purge
module load StdEnv/2023
module load python/3.11
module load gcc arrow

export OPENAI_API_KEY=$(cat ~/chatgpt_api.key)

source venv/bin/activate

mkdir -p logs

DIALECT_NAMES=(
  "Urban African American Vernacular English"
  "Appalachian English"
  "Irish English"
)

DIALECT_SLUGS=(
  "aave"
  "appalachian"
  "irish"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
DIALECT=${DIALECT_NAMES[$TASK_ID]}
DIALECT_SLUG=${DIALECT_SLUGS[$TASK_ID]}

MAX_SAMPLES_ARG=()
if [[ -n "${MAX_SAMPLES:-}" ]]; then
  MAX_SAMPLES_ARG=(--max_samples "$MAX_SAMPLES")
fi

INPUT_CEFR_LEVELS_ARG=()
if [[ -n "${INPUT_CEFR_LEVELS:-}" ]]; then
  INPUT_CEFR_LEVELS_ARG=(--input_cefr_levels "$INPUT_CEFR_LEVELS")
fi

python src/run/main.py \
  --batch_size 5 \
  "${MAX_SAMPLES_ARG[@]}" \
  --text_chunking hybrid \
  --sentence_chunk_min_words 100 \
  --max_chunk_words 80 \
  --max_rules_per_chunk 1 \
  --max_rules_per_row 2 \
  --save_path ./outputs/cefr_texts/dialect \
  --file_name "${DIALECT_SLUG}_gpt41mini_full_hybrid" \
  --input_path ./data/cefr_leveled_texts.csv \
  --text_column text \
  "${INPUT_CEFR_LEVELS_ARG[@]}" \
  --dialect "$DIALECT" \
  --task_name english_dialect \
  --data_path ./ \
  --dataset_name cefr_texts \
  --model_provider openai \
  --model_name gpt-4.1-mini
