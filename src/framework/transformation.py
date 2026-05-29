import copy
import json
import os
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from registry.prompt import *
from utils.guidline_utils import *
from utils import log


def introduces_blank(orig_sentence, transformed_sentence):
    return '<blank>' not in orig_sentence and '<blank>' in transformed_sentence


def _rule_budget_reached(applied_rules, max_rules):
    return max_rules is not None and len(applied_rules) >= max_rules


def _order_guidelines(guideline):
    ordered = list(guideline)
    random.shuffle(ordered)
    return ordered


def _run_in_parallel(items, max_workers, func):
    if max_workers is None or max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(executor.map(func, items))


def _batch_object_field(value, field_name, default=None):
    if isinstance(value, dict):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def _response_content_to_text(content):
    if content is None:
        return ""
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content
    if hasattr(content, "text"):
        return content.text
    if hasattr(content, "read"):
        data = content.read()
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    if hasattr(content, "content"):
        data = content.content
        return data.decode("utf-8") if isinstance(data, bytes) else str(data)
    return str(content)


def _extract_batch_chat_content(response_body):
    try:
        return response_body["choices"][0]["message"]["content"]
    except Exception:
        return None


def _write_batch_jsonl(path, requests):
    with open(path, "w", encoding="utf-8") as f:
        for request in requests:
            f.write(json.dumps(request, ensure_ascii=False) + "\n")


def _run_openai_batch_requests(client, requests, output_dir, poll_interval, label):
    if not requests:
        return {}

    os.makedirs(output_dir, exist_ok=True)
    run_id = f"{label}_{uuid.uuid4().hex[:10]}"
    input_path = os.path.join(output_dir, f"{run_id}_input.jsonl")
    output_path = os.path.join(output_dir, f"{run_id}_output.jsonl")
    error_path = os.path.join(output_dir, f"{run_id}_error.jsonl")
    metadata_path = os.path.join(output_dir, f"{run_id}_batch.json")

    _write_batch_jsonl(input_path, requests)
    with open(input_path, "rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    log(f"OpenAI Batch submitted: label={label}, batch_id={batch.id}, requests={len(requests)}, input={input_path}")

    terminal_statuses = {"completed", "failed", "expired", "cancelled"}
    while _batch_object_field(batch, "status") not in terminal_statuses:
        status = _batch_object_field(batch, "status", "unknown")
        request_counts = _batch_object_field(batch, "request_counts", None)
        log(f"OpenAI Batch polling: label={label}, batch_id={batch.id}, status={status}, request_counts={request_counts}")
        time.sleep(poll_interval)
        batch = client.batches.retrieve(batch.id)

    batch_metadata = dict(batch) if isinstance(batch, dict) else {
        key: _batch_object_field(batch, key)
        for key in (
            "id",
            "endpoint",
            "errors",
            "input_file_id",
            "completion_window",
            "status",
            "output_file_id",
            "error_file_id",
            "created_at",
            "in_progress_at",
            "expires_at",
            "completed_at",
            "failed_at",
            "expired_at",
            "cancelled_at",
            "request_counts",
        )
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(batch_metadata, f, ensure_ascii=False, indent=2, default=str)

    status = _batch_object_field(batch, "status")
    output_file_id = _batch_object_field(batch, "output_file_id")
    error_file_id = _batch_object_field(batch, "error_file_id")

    output_text = ""
    if output_file_id:
        output_text = _response_content_to_text(client.files.content(output_file_id))
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(output_text)

    error_text = ""
    if error_file_id:
        error_text = _response_content_to_text(client.files.content(error_file_id))
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_text)

    if status != "completed":
        raise RuntimeError(
            f"OpenAI Batch {batch.id} ended with status={status}. "
            f"metadata={metadata_path}, output={output_path if output_text else '<none>'}, "
            f"errors={error_path if error_text else '<none>'}"
        )

    results = {}
    for line in output_text.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        custom_id = item.get("custom_id")
        response = item.get("response")
        error = item.get("error")
        if error is not None:
            results[custom_id] = (None, json.dumps(error, ensure_ascii=False))
            continue

        status_code = response.get("status_code") if isinstance(response, dict) else None
        body = response.get("body") if isinstance(response, dict) else None
        if status_code is None or status_code >= 400:
            results[custom_id] = (None, json.dumps(response, ensure_ascii=False))
            continue

        content = _extract_batch_chat_content(body)
        results[custom_id] = (content, None)

    missing_ids = [request["custom_id"] for request in requests if request["custom_id"] not in results]
    for custom_id in missing_ids:
        results[custom_id] = (None, "missing batch output")

    log(f"OpenAI Batch completed: label={label}, batch_id={batch.id}, output={output_path}")
    return results


def _has_transformed_sentence_marker(response):
    if response is None:
        return False
    response = response.lower()
    return "transformed sentence:" in response or "final broken sentence:" in response



def framework_application(guideline, task):
    guideline = guideline[1]

    guideline_instruction, example = extract_guideline_examples(guideline, task)
    system_message = return_system_message(guideline_instruction)

    message = [
        {"role": "user", "content": system_message},
        {"role": "assistant", "content": "Well Understood."},
        {"role": "user", "content": example[0]['input']},
        {"role": "assistant", "content": example[0]['output']}
        ]

    return message



def transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config, max_rules_per_chunk=None):
    """
    sentence (list of string) where list size is equal to batch size
    """

    if type(sentence) is tuple:     # tuple이면 list로 바꾸기
        sentence = list(sentence)

    orig_sentence = copy.deepcopy(sentence)

    whole_responses = [[] for _ in range(len(sentence))]
    applied_rules = [[] for _ in range(len(sentence))]  # rules that are answered yes to all identification questions
    mid_transformed_sentences = [[] for _ in range(len(sentence))] # transformed sentences that are transformed by applied rules
    judge_responses = [[] for _ in range(len(sentence))] # judge response to each transformed sentence
    transformed_sentences = [[] for _ in range(len(sentence))] # final transformed sentence

    # shuffle guideline
    guideline = _order_guidelines(guideline)

    for i in range(len(guideline)):
        feature = guideline[i][0]

        input_prompt = framework_application(guideline=guideline[i], task=task_config.task_name)

        active_indices = [
            idx for idx in range(len(sentence))
            if not _rule_budget_reached(applied_rules[idx], max_rules_per_chunk)
        ]
        if not active_indices:
            continue

        batch_input = [
            input_prompt + [{"role": 'user', "content": f"**Original Sentence:** {sentence[idx]}"}]
            for idx in active_indices
        ]
        chat_batch_input = list()

        for input in batch_input:
            text = tokenizer.apply_chat_template(
                input,
                tokenize=False,
                add_generation_prompt=True
            )
            chat_batch_input.append(text)

        responses = client.completions.create(
            model=model_config.model_name,
            prompt=chat_batch_input,
            **sampling_params
            )
        
        for response_idx, response in enumerate(responses.choices):
            num = active_indices[response_idx]
            # save all responses
            whole_responses[num].append(response.text)

            if response.text is None:
                continue

            transformed_sentence = extract_transformed_sentence(response.text)
            if not transformed_sentence.strip() or ('no change' in transformed_sentence.lower()):
                continue
            if introduces_blank(orig_sentence[num], transformed_sentence):
                continue

            else:
                # save the transformed sentences
                mid_transformed_sentences[num].append(transformed_sentence)

                semantic_input_prompt = semantic_check(orig_sentence[num], transformed_sentence)
                semantic_response = client.chat.completions.create(
                    model=model_config.model_name,
                    messages=[{'role': 'user', 'content': semantic_input_prompt}]
                )

                # save judge response
                judge_responses[num].append(semantic_response.choices[0].message.content.lower())

                if 'no' in semantic_response.choices[0].message.content.lower():
                    sentence[num] = transformed_sentence
                    applied_rules[num].append(feature)
                    transformed_sentences[num].append(transformed_sentence)
        
    iter_result = list()

    for num in range(len(sentence)):
        iter_result.append({
            'orig_sentence': orig_sentence[num],
            'whole_response': whole_responses[num],
            'mid_transformed_sentences': mid_transformed_sentences[num],
            'judge_repsonse': judge_responses[num],
            'applied_rules': applied_rules[num],
            'transformed_sentences': transformed_sentences[num],
            'final_sentence': sentence[num]
        })

    return iter_result



def openai_framework_application(guideline, task):
    guideline = guideline[1]

    guideline_instruction, example = extract_guideline_examples(guideline, task)
    system_message = return_system_message(guideline_instruction)

    message = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": example[0]['input']},
        {"role": "assistant", "content": example[0]['output']}
    ]

    return message



def openai_transformation(
    sentence,
    guideline,
    client,
    sampling_params,
    task_config,
    model_config,
    max_rules_per_chunk=None,
    openai_parallelism=1,
    openai_call_mode="sync",
    openai_batch_output_dir=None,
    openai_batch_poll_interval=60,
):
    """
    Hosted chat-completion transformation.

    sentence: list of strings where list size is equal to batch size.
    """

    if type(sentence) is tuple:
        sentence = list(sentence)
    elif type(sentence) is str:
        sentence = [sentence]

    orig_sentence = copy.deepcopy(sentence)

    whole_responses = [[] for _ in range(len(sentence))]
    applied_rules = [[] for _ in range(len(sentence))]
    mid_transformed_sentences = [[] for _ in range(len(sentence))]
    judge_responses = [[] for _ in range(len(sentence))]
    transformed_sentences = [[] for _ in range(len(sentence))]
    model_errors = [[] for _ in range(len(sentence))]
    semantic_errors = [[] for _ in range(len(sentence))]
    no_change_response_counts = [0 for _ in range(len(sentence))]
    parse_failure_counts = [0 for _ in range(len(sentence))]
    blank_rejection_counts = [0 for _ in range(len(sentence))]
    semantic_rejection_counts = [0 for _ in range(len(sentence))]

    guideline = _order_guidelines(guideline)

    transformation_params = {
        'temperature': sampling_params['temperature'],
        'top_p': sampling_params['top_p'],
        'max_tokens': sampling_params['max_tokens'],
    }
    semantic_model_name = model_config.semantic_model_name or model_config.model_name
    consecutive_model_errors = 0
    max_consecutive_model_errors = 10

    use_batch_api = openai_call_mode == "batch"
    if use_batch_api and not openai_batch_output_dir:
        openai_batch_output_dir = os.path.join(".", "openai_batch")

    for i in range(len(guideline)):
        feature = guideline[i][0]

        input_prompt = openai_framework_application(guideline=guideline[i], task=task_config.task_name)

        active_prompts = [
            (
                num,
                input_prompt + [{"role": 'user', "content": f"**Original Sentence:** {sentence[num]}"}],
            )
            for num in range(len(sentence))
            if not _rule_budget_reached(applied_rules[num], max_rules_per_chunk)
        ]
        if not active_prompts:
            continue

        if use_batch_api:
            batch_requests = []
            batch_lookup = {}
            for request_idx, (num, prompt) in enumerate(active_prompts):
                custom_id = f"transform:{i}:{request_idx}:row:{num}"
                batch_lookup[custom_id] = num
                batch_requests.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_config.model_name,
                        "messages": prompt,
                        **transformation_params,
                    },
                })
            batch_results = _run_openai_batch_requests(
                client,
                batch_requests,
                openai_batch_output_dir,
                openai_batch_poll_interval,
                f"transform_feature_{i}",
            )
            transformation_results = [
                (batch_lookup[custom_id], response, error)
                for custom_id, (response, error) in batch_results.items()
            ]
        else:
            def call_transformation(item):
                num, prompt = item
                try:
                    responses = client.chat.completions.create(
                        model=model_config.model_name,
                        messages=prompt,
                        **transformation_params
                    )
                except Exception as e:
                    return num, None, str(e)

                return num, responses.choices[0].message.content, None

            transformation_results = _run_in_parallel(active_prompts, openai_parallelism, call_transformation)

        candidate_tasks = []
        for num, response, error in transformation_results:
            if error is not None:
                consecutive_model_errors += 1
                model_errors[num].append(error)
                whole_responses[num].append(error)
                if consecutive_model_errors >= max_consecutive_model_errors:
                    raise RuntimeError(
                        f"Hosted OpenAI transformation failed {consecutive_model_errors} times in a row. "
                        f"Latest feature: {feature}. Latest error: {error}"
                    )
                continue

            consecutive_model_errors = 0
            whole_responses[num].append(response)

            if response is None:
                continue

            transformed_sentence = extract_transformed_sentence(response)

            if not transformed_sentence.strip() or ('no change' in transformed_sentence.lower()):
                no_change_response_counts[num] += 1
                if not _has_transformed_sentence_marker(response):
                    parse_failure_counts[num] += 1
                continue
            if introduces_blank(orig_sentence[num], transformed_sentence):
                blank_rejection_counts[num] += 1
                continue

            mid_transformed_sentences[num].append(transformed_sentence)

            semantic_input_prompt = semantic_check(orig_sentence[num], transformed_sentence)
            candidate_tasks.append((num, transformed_sentence, semantic_input_prompt))

        if use_batch_api:
            batch_requests = []
            batch_lookup = {}
            for request_idx, (num, transformed_sentence, semantic_input_prompt) in enumerate(candidate_tasks):
                custom_id = f"semantic:{i}:{request_idx}:row:{num}"
                batch_lookup[custom_id] = (num, transformed_sentence)
                batch_requests.append({
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": semantic_model_name,
                        "messages": [{'role': 'user', 'content': semantic_input_prompt}],
                        "temperature": 0,
                        "max_tokens": 10,
                    },
                })
            batch_results = _run_openai_batch_requests(
                client,
                batch_requests,
                openai_batch_output_dir,
                openai_batch_poll_interval,
                f"semantic_feature_{i}",
            )
            semantic_results = [
                (
                    batch_lookup[custom_id][0],
                    batch_lookup[custom_id][1],
                    response.lower() if response is not None else None,
                    error.lower() if error is not None else None,
                )
                for custom_id, (response, error) in batch_results.items()
            ]
        else:
            def call_semantic_check(item):
                num, transformed_sentence, semantic_input_prompt = item
                try:
                    semantic_response = client.chat.completions.create(
                        model=semantic_model_name,
                        messages=[{'role': 'user', 'content': semantic_input_prompt}],
                        temperature=0,
                        max_tokens=10,
                    )
                except Exception as e:
                    return num, transformed_sentence, None, str(e).lower()

                return num, transformed_sentence, semantic_response.choices[0].message.content.lower(), None

            semantic_results = _run_in_parallel(candidate_tasks, openai_parallelism, call_semantic_check)
        for num, transformed_sentence, judge_response, error in semantic_results:
            if error is not None:
                semantic_errors[num].append(error)
                judge_responses[num].append(error)
                continue

            judge_responses[num].append(judge_response)

            if 'no' in judge_response:
                sentence[num] = transformed_sentence
                applied_rules[num].append(feature)
                transformed_sentences[num].append(transformed_sentence)
            else:
                semantic_rejection_counts[num] += 1
    
    iter_result = list()

    for num in range(len(sentence)):
        iter_result.append({
            'orig_sentence': orig_sentence[num],
            'whole_response': whole_responses[num],
            'mid_transformed_sentences': mid_transformed_sentences[num],
            'judge_repsonse': judge_responses[num],
            'applied_rules': applied_rules[num],
            'transformed_sentences': transformed_sentences[num],
            'final_sentence': sentence[num],
            'model_errors': model_errors[num],
            'semantic_errors': semantic_errors[num],
            'model_error_count': len(model_errors[num]),
            'semantic_error_count': len(semantic_errors[num]),
            'no_change_response_count': no_change_response_counts[num],
            'parse_failure_count': parse_failure_counts[num],
            'blank_rejection_count': blank_rejection_counts[num],
            'semantic_rejection_count': semantic_rejection_counts[num],
        })

    return iter_result
