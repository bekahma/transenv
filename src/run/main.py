import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import re
import numpy as np
from tqdm import tqdm
from collections import defaultdict

from configs.parse_arguments import parse_args
from framework.guideline import return_guideline
from framework.data_return import return_dataloader
from framework.transformation import transformation, openai_transformation
from registry.framework import QUESTION_KEY_ID
from utils import log, colorstr
from utils.common import save_func
from utils.cefr_texts import (
    INTERNAL_EMPTY_TEXT_COLUMN,
    aggregate_chunk_results,
    empty_transformation_result,
    load_cefr_text_dataset,
    split_text_chunks,
)
from utils.model_utils import return_model, uses_hosted_openai
from utils.filesys_utils import pickle_load, pickle_save

choice_transform_dataset = []


def _as_batch_list(value):
    if hasattr(value, 'tolist'):
        return value.tolist()
    if type(value) is tuple:
        return list(value)
    if type(value) is list:
        return value
    return [value]


def _batch_dataset(dataset, batch_size):
    column_names = dataset.column_names
    for start_idx in range(0, len(dataset), batch_size):
        rows = dataset[start_idx:start_idx + batch_size]
        yield {column: rows[column] for column in column_names}


def _torch_dataloader(dataset, batch_size):
    from torch.utils.data import DataLoader

    return DataLoader(dataset, batch_size, shuffle=False)


def _load_tokenizer(model_config):
    from transformers import AutoTokenizer

    tokenizer_name = model_config.tokenizer or model_config.model_name
    return AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=os.environ.get("MODEL_DIR", None))


def _load_dataset(*args, **kwargs):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError("The Hugging Face datasets package is required. Install it with `pip install -r requirements-openai.txt`.") from exc

    return load_dataset(*args, **kwargs)


def _transform_sentences(sentences, guideline, client, tokenizer, sampling_params, task_config, model_config, use_hosted_openai, batch_size, max_rules_per_chunk=None):
    results = []

    for start_idx in range(0, len(sentences), batch_size):
        batch = sentences[start_idx:start_idx + batch_size]
        if use_hosted_openai:
            batch_results = openai_transformation(batch, guideline, client, sampling_params, task_config, model_config, max_rules_per_chunk=max_rules_per_chunk)
        else:
            batch_results = transformation(batch, guideline, client, tokenizer, sampling_params, task_config, model_config, max_rules_per_chunk=max_rules_per_chunk)
        results.extend(batch_results)

    return results


def _get_applied_rules(result):
    return list(result.get("applied_rules", result.get("applied_rule", [])) or [])


def _count_rule_applications(result):
    return len(_get_applied_rules(result))


def _min_rule_limit(*limits):
    active_limits = [limit for limit in limits if limit is not None]
    if not active_limits:
        return None
    return min(active_limits)


def _limit_result_to_rule_budget(result, chunk_text, max_rules):
    if max_rules is None:
        return result

    rules = _get_applied_rules(result)
    if len(rules) <= max_rules:
        return result
    if max_rules <= 0:
        return empty_transformation_result(chunk_text)

    transformed_sentences = result.get("transformed_sentences", [])
    if len(transformed_sentences) < max_rules:
        return empty_transformation_result(chunk_text)

    limited_result = dict(result)
    limited_rules = rules[:max_rules]
    limited_transformed_sentences = transformed_sentences[:max_rules]

    limited_result["applied_rules"] = limited_rules
    if "applied_rule" in limited_result:
        limited_result["applied_rule"] = limited_rules
    limited_result["transformed_sentences"] = limited_transformed_sentences
    limited_result["final_sentence"] = limited_transformed_sentences[-1]
    limited_result["rule_limit_truncated"] = True

    return limited_result


def _trim_results_to_row_rule_budget(chunk_results, chunks, max_rules_per_row):
    if max_rules_per_row is None:
        return chunk_results

    remaining_rules = max_rules_per_row
    trimmed_results = []

    for idx, result in enumerate(chunk_results):
        if remaining_rules <= 0:
            trimmed_results.append(empty_transformation_result(chunks[idx]["text"]))
            continue

        limited_result = _limit_result_to_rule_budget(result, chunks[idx]["text"], remaining_rules)
        remaining_rules -= _count_rule_applications(limited_result)
        trimmed_results.append(limited_result)

    return trimmed_results


def _transform_cefr_chunks(chunks, guideline, client, tokenizer, sampling_params, task_config, model_config, use_hosted_openai, batch_size, max_rules_per_chunk=None, max_rules_per_row=None):
    chunk_sentences = [chunk["text"] for chunk in chunks]

    if max_rules_per_row is None:
        return _transform_sentences(
            chunk_sentences,
            guideline,
            client,
            tokenizer,
            sampling_params,
            task_config,
            model_config,
            use_hosted_openai,
            batch_size,
            max_rules_per_chunk=max_rules_per_chunk,
        )

    chunk_results = []
    remaining_rules = max_rules_per_row

    for chunk in chunks:
        if remaining_rules <= 0:
            chunk_results.append(empty_transformation_result(chunk["text"]))
            continue

        chunk_rule_budget = _min_rule_limit(max_rules_per_chunk, remaining_rules)
        result = _transform_sentences(
            [chunk["text"]],
            guideline,
            client,
            tokenizer,
            sampling_params,
            task_config,
            model_config,
            use_hosted_openai,
            1,
            max_rules_per_chunk=chunk_rule_budget,
        )[0]
        result = _limit_result_to_rule_budget(result, chunk["text"], remaining_rules)
        remaining_rules -= _count_rule_applications(result)
        chunk_results.append(result)

    return chunk_results


def _validate_generation_limits(generation_config):
    for name in ("max_samples", "max_rules_per_chunk", "max_rules_per_row"):
        value = getattr(generation_config, name, None)
        if value is not None and value < 0:
            raise ValueError(f"--{name} must be greater than or equal to 0")



def main():
    # Initialize arguments
    generation_config, model_config, dataset_config, task_config, save_config = parse_args()
    _validate_generation_limits(generation_config)

    if dataset_config.dataset_name is None:
        raise AssertionError(colorstr('red', 'Dataset name should be specified!'))
    
    if task_config.task_name == 'L1' and task_config.cefr_level is None:
        raise AssertionError(colorstr('red', 'You should specify cefr level in order to change L1.'))

    if task_config.task_name == 'L1':
        log(f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, l1: {colorstr(task_config.l1)}, cefr: {colorstr(task_config.cefr_level)}, Rerun: {colorstr(bool(generation_config.rerun))}')
    elif task_config.task_name == 'english_dialect':
        log(f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, dialect: {colorstr(task_config.dialect)}, Rerun: {colorstr(bool(generation_config.rerun))}')
    elif task_config.task_name == 'cefr':
        log(f'Dataset: {colorstr(dataset_config.dataset_name)}, Task: {colorstr(task_config.task_name)}, CEFR level: {colorstr(task_config.cefr_level)}, Rerun: {colorstr(bool(generation_config.rerun))}', )

    os.makedirs(save_config.save_path, exist_ok=True)

    if dataset_config.sampling is True:
        save_config.file_name += '_sampling'

    # Intialize model
    client = return_model(model_config=model_config)
    use_hosted_openai = uses_hosted_openai(model_config)
    tokenizer = None
    if use_hosted_openai is False:
        tokenizer = _load_tokenizer(model_config)

    # Guideline
    guideline = return_guideline(task_config=task_config, dataset_name=dataset_config.dataset_name, data_path=save_config.data_path)

    to_save = list()
    to_save_choice = defaultdict(list)

    # Resume
    start_idx = 0
    if os.path.exists(os.path.join(save_config.save_path, f'{save_config.file_name}.pk')):
        log('Found existing file! Loading progress...')
        resume_dict = pickle_load(os.path.join(save_config.save_path, f'{save_config.file_name}.pk'))
        to_save = resume_dict['question']
        start_idx = len(to_save)

    # Dataloader
    if dataset_config.dataset_name == 'cefr_texts':
        dataset = load_cefr_text_dataset(
            dataset_config=dataset_config,
            generation_config=generation_config,
            start_idx=start_idx,
        )
        if len(dataset) == 0 and start_idx:
            log(f'No remaining cefr_texts rows to process; saving {len(to_save)} resumed rows.')
            to_save_dict = {'question': to_save}
            save_func(to_save_dict, save_config, dataset_config, generation_config, task_config)
            return
        dataloader = _batch_dataset(dataset, generation_config.batch_size)

    elif task_config.task_name == 'cefr':
        dataset = _load_dataset(
            "csv",
            data_files = {"test": f'{save_config.data_path}/assets/vocab_processed/{dataset_config.dataset_name}_{(task_config.cefr_level).lower()}.csv'},
            split="test",
        )

        if generation_config.rerun is not None:
            rerun_index = list(np.load(generation_config.rerun))
            dataset = dataset.select(rerun_index)

        dataloader = _torch_dataloader(dataset, generation_config.batch_size)


    elif task_config.task_name == 'L1':
        cefr_data_path = ('/').join(save_config.save_path.split('/')[:-2])

        dataset = _load_dataset(
            "csv",  
            data_files={"test": f'{cefr_data_path}/assets/cefr/{dataset_config.dataset_name}/{task_config.cefr_level}.csv'},
            split='test',
        )

        if generation_config.rerun is not None:
            rerun_index = list(np.load(generation_config.rerun))
            dataset = dataset.select(rerun_index)
    
        dataloader = _torch_dataloader(dataset, generation_config.batch_size)

    elif task_config.task_name == 'english_dialect':
        dataloader = return_dataloader(dataset_config=dataset_config, generation_config=generation_config, start_idx=start_idx)


    # Sampling Parameters
    sampling_params = {
        'temperature': generation_config.temperature,
        'top_p': generation_config.top_p,
        'max_tokens': generation_config.max_tokens,
    }
    
    for it, sample in enumerate(tqdm(dataloader)):
        # Question
        sentence = sample[QUESTION_KEY_ID[dataset_config.dataset_name]]

        if dataset_config.dataset_name == 'cefr_texts':
            sentence = _as_batch_list(sentence)
            empty_mask = _as_batch_list(sample[INTERNAL_EMPTY_TEXT_COLUMN])
            transformed_batch = [None for _ in range(len(sentence))]

            for idx, value in enumerate(sentence):
                if empty_mask[idx]:
                    transformed_batch[idx] = empty_transformation_result(value)
                    continue

                chunks = split_text_chunks(
                    value,
                    mode=dataset_config.text_chunking,
                    max_chunk_words=dataset_config.max_chunk_words,
                    sentence_chunk_min_words=dataset_config.sentence_chunk_min_words,
                )

                if not chunks:
                    transformed_batch[idx] = empty_transformation_result(value)
                    continue

                chunk_results = _transform_cefr_chunks(
                    chunks,
                    guideline,
                    client,
                    tokenizer,
                    sampling_params,
                    task_config,
                    model_config,
                    use_hosted_openai,
                    generation_config.batch_size,
                    max_rules_per_chunk=generation_config.max_rules_per_chunk,
                    max_rules_per_row=generation_config.max_rules_per_row,
                )
                transformed_batch[idx] = aggregate_chunk_results(value, chunks, chunk_results)

            for idx, value in enumerate(sentence):
                if transformed_batch[idx] is None:
                    transformed_batch[idx] = empty_transformation_result(value)

            iter_result = transformed_batch
        else:
            sentence = [re.sub(r'_{2,}', '<blank>', s) for s in sentence]

            if use_hosted_openai:
                iter_result = openai_transformation(sentence, guideline, client, sampling_params, task_config, model_config, max_rules_per_chunk=generation_config.max_rules_per_chunk)
            else:
                iter_result = transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config, max_rules_per_chunk=generation_config.max_rules_per_chunk)

        to_save.extend(iter_result)

        if dataset_config.dataset_name in choice_transform_dataset:

            # choices transform
            for choice_num, sentence in enumerate(sample['choices']['text']):
                iter_result = transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config, max_rules_per_chunk=generation_config.max_rules_per_chunk)
                to_save_choice[choice_num].extend(iter_result)

            to_save_dict = {
                'question': to_save,
                'choices': to_save_choice
            }

        else:
            to_save_dict = {'question': to_save}

        if generation_config.rerun is None:
            pickle_save(os.path.join(save_config.save_path, f'{save_config.file_name}.pk'), to_save_dict)
        elif generation_config.rerun is not None:
            pickle_save(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.pk'), to_save_dict)
        
        save_func(to_save_dict, save_config, dataset_config, generation_config, task_config)        




if __name__ == "__main__":
    main()
