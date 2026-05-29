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

# export OPENAI_API_KEY=$(cat ~/chatgpt_api.key)

set -a
source .env
set +a

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

INPUT_CEFR_LEVELS=${INPUT_CEFR_LEVELS:-A1,A2}
INPUT_CEFR_LEVELS_ARG=()
if [[ -n "${INPUT_CEFR_LEVELS:-}" && "$INPUT_CEFR_LEVELS" != "none" ]]; then
  INPUT_CEFR_LEVELS_ARG=(--input_cefr_levels "$INPUT_CEFR_LEVELS")
fi

MAX_RULES_PER_CHUNK=${MAX_RULES_PER_CHUNK:-3}
MAX_RULES_PER_ROW=${MAX_RULES_PER_ROW:-3}
OPENAI_PARALLELISM=${OPENAI_PARALLELISM:-2}
TEXT_CHUNKING=${TEXT_CHUNKING:-row}
SENTENCE_CHUNK_MIN_WORDS=${SENTENCE_CHUNK_MIN_WORDS:-100}
MAX_CHUNK_WORDS=${MAX_CHUNK_WORDS:-80}
RUN_SUFFIX=${RUN_SUFFIX:-a1a2_probe}
FILE_NAME="${DIALECT_SLUG}_gpt41mini_${TEXT_CHUNKING}_${RUN_SUFFIX}"
WRITE_CAA_PAIRS=${WRITE_CAA_PAIRS:-1}
WRITE_CAA_PAIRS_ARG=()
if [[ "$WRITE_CAA_PAIRS" != "0" && "$WRITE_CAA_PAIRS" != "false" && "$WRITE_CAA_PAIRS" != "False" ]]; then
  WRITE_CAA_PAIRS_ARG=(--write_caa_pairs)
fi

echo "Dialect: $DIALECT"
echo "File suffix: $RUN_SUFFIX"
echo "File name: $FILE_NAME"
echo "MAX_SAMPLES: ${MAX_SAMPLES:-<none>}"
echo "INPUT_CEFR_LEVELS: ${INPUT_CEFR_LEVELS:-<none>}"
echo "MAX_RULES_PER_CHUNK: $MAX_RULES_PER_CHUNK"
echo "MAX_RULES_PER_ROW: $MAX_RULES_PER_ROW"
echo "OPENAI_PARALLELISM: $OPENAI_PARALLELISM"
echo "TEXT_CHUNKING: $TEXT_CHUNKING"
echo "SENTENCE_CHUNK_MIN_WORDS: $SENTENCE_CHUNK_MIN_WORDS"
echo "MAX_CHUNK_WORDS: $MAX_CHUNK_WORDS"
echo "WRITE_CAA_PAIRS: $WRITE_CAA_PAIRS"

python src/run/main.py \
  --batch_size 5 \
  "${MAX_SAMPLES_ARG[@]}" \
  --text_chunking "$TEXT_CHUNKING" \
  --sentence_chunk_min_words "$SENTENCE_CHUNK_MIN_WORDS" \
  --max_chunk_words "$MAX_CHUNK_WORDS" \
  --max_rules_per_chunk "$MAX_RULES_PER_CHUNK" \
  --max_rules_per_row "$MAX_RULES_PER_ROW" \
  "${WRITE_CAA_PAIRS_ARG[@]}" \
  --openai_parallelism "$OPENAI_PARALLELISM" \
  --save_path ./outputs/cefr_texts/dialect \
  --file_name "$FILE_NAME" \
  --input_path ./data/cefr_leveled_texts.csv \
  --text_column text \
  "${INPUT_CEFR_LEVELS_ARG[@]}" \
  --dialect "$DIALECT" \
  --task_name english_dialect \
  --data_path ./ \
  --dataset_name cefr_texts \
  --model_provider openai \
  --model_name gpt-4.1-mini
