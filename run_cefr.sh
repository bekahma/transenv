#!/bin/bash
#SBATCH --job-name=cefr_l1
#SBATCH --account=aip-creager
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

module purge
module load StdEnv/2023
module load python/3.11
module load gcc arrow

export OPENAI_API_KEY=$(cat ~/chatgpt_api.key)

source venv/bin/activate

mkdir -p logs

python src/run/main.py \
  --batch_size 5 \
  --max_samples 10 \
  --save_path ./outputs/cefr_texts/l1 \
  --file_name A_arabic_gpt41mini_10_v2 \
  --input_path ./data/cefr_leveled_texts.csv \
  --text_column text \
  --input_cefr_levels A1,A2 \
  --l1 Arabic \
  --task_name L1 \
  --cefr_level A \
  --data_path ./ \
  --dataset_name cefr_texts \
  --model_provider openai \
  --model_name gpt-4.1-mini
