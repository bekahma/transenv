import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import re
import time
import numpy as np
from tqdm import tqdm
from collections import Counter, defaultdict

from configs.parse_arguments import parse_args
from framework.guideline import return_guideline
from framework.data_return import return_dataloader
from framework.transformation import transformation, openai_transformation
from registry.framework import QUESTION_KEY_ID
from utils import log, colorstr
from utils.common import save_func
from utils.cefr_texts import (
    DIAGNOSTIC_COUNT_KEYS,
    INTERNAL_EMPTY_TEXT_COLUMN,
    INTERNAL_ROW_INDEX_COLUMN,
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


def _transform_sentences(sentences, guideline, client, tokenizer, sampling_params, task_config, model_config, use_hosted_openai, batch_size, max_rules_per_chunk=None, openai_parallelism=1):
    results = []

    for start_idx in range(0, len(sentences), batch_size):
        batch = sentences[start_idx:start_idx + batch_size]
        if use_hosted_openai:
            batch_results = openai_transformation(
                batch,
                guideline,
                client,
                sampling_params,
                task_config,
                model_config,
                max_rules_per_chunk=max_rules_per_chunk,
                openai_parallelism=openai_parallelism,
            )
        else:
            batch_results = transformation(
                batch,
                guideline,
                client,
                tokenizer,
                sampling_params,
                task_config,
                model_config,
                max_rules_per_chunk=max_rules_per_chunk,
            )
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


def _transform_cefr_chunks_in_parallel_windows(chunks, guideline, client, tokenizer, sampling_params, task_config, model_config, use_hosted_openai, batch_size, max_rules_per_chunk=None, max_rules_per_row=None, openai_parallelism=1):
    chunk_results = []
    remaining_rules = max_rules_per_row
    window_size = max(1, min(batch_size, openai_parallelism))
    start_idx = 0

    while start_idx < len(chunks):
        if remaining_rules <= 0:
            chunk_results.extend(
                empty_transformation_result(chunk["text"])
                for chunk in chunks[start_idx:]
            )
            break

        window_chunks = chunks[start_idx:start_idx + window_size]
        chunk_rule_budget = _min_rule_limit(max_rules_per_chunk, remaining_rules)

        window_results = _transform_sentences(
            [chunk["text"] for chunk in window_chunks],
            guideline,
            client,
            tokenizer,
            sampling_params,
            task_config,
            model_config,
            use_hosted_openai,
            len(window_chunks),
            max_rules_per_chunk=chunk_rule_budget,
            openai_parallelism=openai_parallelism,
        )
        window_results = _trim_results_to_row_rule_budget(window_results, window_chunks, remaining_rules)

        remaining_rules -= sum(_count_rule_applications(result) for result in window_results)
        chunk_results.extend(window_results)
        start_idx += len(window_chunks)

    return chunk_results


def _transform_cefr_chunks(chunks, guideline, client, tokenizer, sampling_params, task_config, model_config, use_hosted_openai, batch_size, max_rules_per_chunk=None, max_rules_per_row=None, openai_parallelism=1):
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
            openai_parallelism=openai_parallelism,
        )

    if use_hosted_openai and openai_parallelism is not None and openai_parallelism > 1:
        return _transform_cefr_chunks_in_parallel_windows(
            chunks,
            guideline,
            client,
            tokenizer,
            sampling_params,
            task_config,
            model_config,
            use_hosted_openai,
            batch_size,
            max_rules_per_chunk=max_rules_per_chunk,
            max_rules_per_row=max_rules_per_row,
            openai_parallelism=openai_parallelism,
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
            openai_parallelism=openai_parallelism,
        )[0]
        result = _limit_result_to_rule_budget(result, chunk["text"], remaining_rules)
        remaining_rules -= _count_rule_applications(result)
        chunk_results.append(result)

    return chunk_results


def _validate_generation_limits(generation_config):
    if getattr(generation_config, "batch_size", 1) <= 0:
        raise ValueError("--batch_size must be greater than 0")

    if getattr(generation_config, "openai_parallelism", 1) <= 0:
        raise ValueError("--openai_parallelism must be greater than 0")

    max_samples = getattr(generation_config, "max_samples", None)
    if max_samples is not None and max_samples < 0:
        raise ValueError("--max_samples must be greater than or equal to 0")

    for name in ("max_rules_per_chunk", "max_rules_per_row"):
        value = getattr(generation_config, name, None)
        if value is not None and value <= 0:
            raise ValueError(f"--{name} must be greater than 0 when provided")


def _first_nonempty(values):
    for value in values:
        if value:
            return str(value)
    return ""


def _short_text(value, max_chars=300):
    value = str(value or "").replace("\n", "\\n")
    if len(value) <= max_chars:
        return value
    return value[:max_chars - 3] + "..."


def _summarize_outputs(outputs):
    summary = Counter()
    summary["rows"] = len(outputs)

    for output in outputs:
        orig_sentence = output.get("orig_sentence", "")
        final_sentence = output.get("final_sentence", orig_sentence)
        summary["changed_rows"] += int(final_sentence != orig_sentence)
        summary["applied_rules"] += len(_get_applied_rules(output))
        summary["model_response_count"] += len(output.get("whole_response", []))
        summary["candidate_transform_count"] += len(output.get("mid_transformed_sentences", []))
        summary["semantic_judge_count"] += len(output.get("judge_repsonse", []))
        summary["chunk_count"] += output.get("chunk_count", 1)
        for key in DIAGNOSTIC_COUNT_KEYS:
            summary[key] += output.get(key, 0)

    return summary


def _log_cefr_row_summary(row_label, output, elapsed_seconds):
    summary = _summarize_outputs([output])
    message = (
        f"cefr_texts row {row_label}: chunks={summary['chunk_count']}, "
        f"responses={summary['model_response_count']}, candidates={summary['candidate_transform_count']}, "
        f"semantic_checks={summary['semantic_judge_count']}, accepted_rules={summary['applied_rules']}, "
        f"errors={summary['model_error_count']}, no_change={summary['no_change_response_count']}, "
        f"parse_failures={summary['parse_failure_count']}, elapsed={elapsed_seconds:.1f}s"
    )

    if summary["applied_rules"]:
        log(message)
        return

    sample_error = _first_nonempty(output.get("model_errors", []))
    sample_response = _first_nonempty(output.get("whole_response", []))
    detail = sample_error or sample_response
    if detail:
        message = f"{message}; sample={'error' if sample_error else 'response'}: {_short_text(detail)}"
    log(message, level="warning")


def _raise_if_all_model_calls_failed(row_label, output):
    model_responses = len(output.get("whole_response", []))
    model_errors = output.get("model_error_count", 0)
    if model_responses == 0:
        return
    if model_errors < model_responses:
        return
    if model_errors < 5:
        return

    sample_error = _short_text(_first_nonempty(output.get("model_errors", [])), max_chars=1000)
    raise RuntimeError(
        f"All hosted OpenAI transformation calls failed for cefr_texts row {row_label} "
        f"({model_errors}/{model_responses} errors). First error: {sample_error}"
    )


def _transform_cefr_row_batch(sentences, empty_mask, row_labels, guideline, client, tokenizer, sampling_params, task_config, model_config, use_hosted_openai, generation_config):
    transformed_batch = [None for _ in range(len(sentences))]
    active_indices = []

    for idx, value in enumerate(sentences):
        if empty_mask[idx]:
            transformed_batch[idx] = empty_transformation_result(value)
            log(f"cefr_texts row {row_labels[idx]}: empty text; skipped transformation.")
        else:
            active_indices.append(idx)

    if not active_indices:
        return transformed_batch

    row_rule_budget = _min_rule_limit(
        generation_config.max_rules_per_chunk,
        generation_config.max_rules_per_row,
    )
    batch_start = time.monotonic()
    log(
        f"cefr_texts row-chunk batch: transforming {len(active_indices)} rows "
        f"with up to {len(guideline)} feature checks per row before early stopping."
    )

    row_results = _transform_sentences(
        [sentences[idx] for idx in active_indices],
        guideline,
        client,
        tokenizer,
        sampling_params,
        task_config,
        model_config,
        use_hosted_openai,
        generation_config.batch_size,
        max_rules_per_chunk=row_rule_budget,
        openai_parallelism=generation_config.openai_parallelism,
    )
    elapsed_seconds = time.monotonic() - batch_start

    for result_idx, row_idx in enumerate(active_indices):
        chunk = [{"text": sentences[row_idx], "separator": ""}]
        output = aggregate_chunk_results(sentences[row_idx], chunk, [row_results[result_idx]])
        transformed_batch[row_idx] = output
        _log_cefr_row_summary(row_labels[row_idx], output, elapsed_seconds)
        _raise_if_all_model_calls_failed(row_labels[row_idx], output)

    return transformed_batch



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
    log(f'Loaded {colorstr(len(guideline))} transformation guidelines.')

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
        log(
            "Generation controls: "
            f"batch_size={generation_config.batch_size}, "
            f"openai_parallelism={generation_config.openai_parallelism}, "
            f"text_chunking={dataset_config.text_chunking}, "
            f"max_rules_per_chunk={generation_config.max_rules_per_chunk}, "
            f"max_rules_per_row={generation_config.max_rules_per_row}"
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
        batch_start = time.monotonic()
        # Question
        sentence = sample[QUESTION_KEY_ID[dataset_config.dataset_name]]

        if dataset_config.dataset_name == 'cefr_texts':
            sentence = _as_batch_list(sentence)
            empty_mask = _as_batch_list(sample[INTERNAL_EMPTY_TEXT_COLUMN])
            row_indices = _as_batch_list(sample.get(INTERNAL_ROW_INDEX_COLUMN, []))
            row_labels = [
                row_indices[idx] if idx < len(row_indices) else start_idx + len(to_save) + idx
                for idx in range(len(sentence))
            ]

            if dataset_config.text_chunking == "row":
                transformed_batch = _transform_cefr_row_batch(
                    sentence,
                    empty_mask,
                    row_labels,
                    guideline,
                    client,
                    tokenizer,
                    sampling_params,
                    task_config,
                    model_config,
                    use_hosted_openai,
                    generation_config,
                )
            else:
                transformed_batch = [None for _ in range(len(sentence))]

            for idx, value in enumerate(sentence):
                if transformed_batch[idx] is not None:
                    continue

                row_label = row_labels[idx]
                row_start = time.monotonic()
                if empty_mask[idx]:
                    transformed_batch[idx] = empty_transformation_result(value)
                    log(f"cefr_texts row {row_label}: empty text; skipped transformation.")
                    continue

                chunks = split_text_chunks(
                    value,
                    mode=dataset_config.text_chunking,
                    max_chunk_words=dataset_config.max_chunk_words,
                    sentence_chunk_min_words=dataset_config.sentence_chunk_min_words,
                )

                if not chunks:
                    transformed_batch[idx] = empty_transformation_result(value)
                    log(f"cefr_texts row {row_label}: no non-empty chunks; skipped transformation.")
                    continue

                log(
                    f"cefr_texts row {row_label}: starting transformation with "
                    f"{len(chunks)} chunks and up to {len(chunks) * len(guideline)} feature checks "
                    f"before early stopping."
                )
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
                    openai_parallelism=generation_config.openai_parallelism,
                )
                transformed_batch[idx] = aggregate_chunk_results(value, chunks, chunk_results)
                _log_cefr_row_summary(row_label, transformed_batch[idx], time.monotonic() - row_start)
                _raise_if_all_model_calls_failed(row_label, transformed_batch[idx])

            for idx, value in enumerate(sentence):
                if transformed_batch[idx] is None:
                    transformed_batch[idx] = empty_transformation_result(value)

            iter_result = transformed_batch
        else:
            sentence = [re.sub(r'_{2,}', '<blank>', s) for s in sentence]

            if use_hosted_openai:
                iter_result = openai_transformation(
                    sentence,
                    guideline,
                    client,
                    sampling_params,
                    task_config,
                    model_config,
                    max_rules_per_chunk=generation_config.max_rules_per_chunk,
                    openai_parallelism=generation_config.openai_parallelism,
                )
            else:
                iter_result = transformation(
                    sentence,
                    guideline,
                    client,
                    tokenizer,
                    sampling_params,
                    task_config,
                    model_config,
                    max_rules_per_chunk=generation_config.max_rules_per_chunk,
                )

        to_save.extend(iter_result)
        batch_summary = _summarize_outputs(iter_result)
        log(
            f"Batch {it + 1} complete in {time.monotonic() - batch_start:.1f}s: "
            f"rows={batch_summary['rows']}, changed_rows={batch_summary['changed_rows']}, "
            f"accepted_rules={batch_summary['applied_rules']}, responses={batch_summary['model_response_count']}, "
            f"candidates={batch_summary['candidate_transform_count']}, semantic_checks={batch_summary['semantic_judge_count']}, "
            f"model_errors={batch_summary['model_error_count']}, no_change={batch_summary['no_change_response_count']}, "
            f"parse_failures={batch_summary['parse_failure_count']}"
        )

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
