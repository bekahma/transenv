# A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties
This repository is the official implementation of Trans-EnV.
<p align="center">
    <img src="docs/figs/figure1.png"/>
</p>

&nbsp;

&nbsp;



## Requirements 🛠️
To install requirements:
```bash
# Lightweight requirements for local CSV transformation with hosted OpenAI models
pip install -r requirements-openai.txt

# PyTorch install (CUDA 12.1)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Full requirements for local vLLM/Gemini evaluation workflows
pip install -r requirements.txt
```

Then, please set the necessary OS variables in your `.env` file.
```bash
GOOGLE_API_KEY=${GCP_API_KEY}  # For GCP Gemini API call
OPENAI_API_KEY=${OPENAI_API_KEY} # For OpenAI model API call
DATA_DIR=${HF_BENCHMARK_PATH}  # If you don't set, path will be set automatically `/home/${user}/.cache/huggingface`
MODEL_DIR=${HF_MODEL_PATH}  # If you don't set, path will be set automatically `/home/${user}/.cache/huggingface`
```

&nbsp;

&nbsp;


## Execution 🚀
### Trans-EnV
```bash
# Convert MMLU to ESL variety via Trans-EnV
python src/run/main.py --batch_size 15  --save_path ./outputs/mmlu/l1 --file_name A_arabic --l1 Arabic --task_name L1 --cefr_level A --port_num 6001 --dataset_name mmlu --model_name google/gemma-2-27b-it --tokenizer google/gemma-2-27b-it

# Convert a local CEFR-levelled English text CSV to an ESL variety
python src/run/main.py --batch_size 15 --save_path ./outputs/cefr_texts/l1 --file_name A_arabic --input_path ./data/cefr_texts_a.csv --text_column text --input_cefr_levels A1,A2 --l1 Arabic --task_name L1 --cefr_level A --data_path ./ --port_num 6001 --dataset_name cefr_texts --model_name google/gemma-2-27b-it --tokenizer google/gemma-2-27b-it

# Cost-conscious CEFR dialect smoke test with hosted OpenAI.
# Row chunking transforms each text as one unit, allows up to 3 eWAVE features,
# and writes both the raw CSV and a *_caa_pairs.csv file.
MAX_SAMPLES=20 RUN_SUFFIX=smoke20 sbatch --array=0 run_cefr.sh

# Full A1/A2 probe for all three configured dialects.
RUN_SUFFIX=a1a2_row sbatch --array=0-2 run_cefr.sh

# Equivalent direct command for one dialect without Slurm.
python src/run/main.py --batch_size 5 --openai_parallelism 2 --max_samples 20 --text_chunking row --max_rules_per_chunk 3 --max_rules_per_row 3 --write_caa_pairs --save_path ./outputs/cefr_texts/dialect --file_name aave_gpt41mini_row_smoke20 --input_path ./data/cefr_leveled_texts.csv --text_column text --input_cefr_levels A1,A2 --dialect "Urban African American Vernacular English" --task_name english_dialect --data_path ./ --dataset_name cefr_texts --model_provider openai --model_name gpt-4.1-mini

# Rebuild the CAA pair export from an existing raw CSV if needed.
python src/run/filter_cefr_outputs.py --caa_pairs --input_csv ./outputs/cefr_texts/dialect/aave_gpt41mini_row_smoke20.csv --output_dir ./outputs/cefr_texts/dialect/filtered --file_prefix aave_gpt41mini_row_smoke20 --dialect "Urban African American Vernacular English" --transform_model gpt-4.1-mini --semantic_model gpt-4.1-mini --max_edit_rate 0.50 --min_length_ratio 0.50 --max_length_ratio 1.80
```

&nbsp;

### LLM Evaluation for English Varieties
```bash
# LLM perforamnce evaluation for dialect variety of GSM8K
python src/run/benchmark_eval.py --model models/gemini-2.5-pro-preview-03-25 --data-path variety_examples/gsm8k/dialect/aave_rerun.csv --output-dir outputs

# LLM perforamnce evaluation for ESL (L1) variety of GSM8K
python src/run/benchmark_eval.py --model models/gemini-2.5-pro-preview-03-25 --data-path variety_examples/gsm8k/l1/A_arabic_rerun.csv --output-dir outputs
```

&nbsp;

&nbsp;


## Results 📚
Comprehensive summary of LLMs' performance across Standard American English (SAE) and 38 benchmark variants.
The results highlight that most LLMs perform best on tasks in SAE.
<p align="center">
    <img src="docs/figs/figure2.png"/>
</p>

&nbsp;

## License 🔑
This project is licensed under the [MIT License](LICENSE).
You are free to use, modify, and distribute this software with proper attribution.
