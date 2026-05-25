#!/bin/bash
#SBATCH --job-name=cefr_filter
#SBATCH --account=aip-creager
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

module purge
module load StdEnv/2023
module load python/3.11

source venv/bin/activate

mkdir -p logs

INPUT_CSV=${INPUT_CSV:-./outputs/cefr_texts/l1/A_arabic_gpt41mini_10_hybrid.csv}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/cefr_texts/l1/filtered}
FILE_PREFIX=${FILE_PREFIX:-$(basename "${INPUT_CSV}" .csv)}
MAX_EDIT_RATE=${MAX_EDIT_RATE:-0.10}

mkdir -p "${OUTPUT_DIR}"

python src/run/filter_cefr_outputs.py \
  --input_csv "${INPUT_CSV}" \
  --output_dir "${OUTPUT_DIR}" \
  --file_prefix "${FILE_PREFIX}" \
  --max_edit_rate "${MAX_EDIT_RATE}"
