import copy
import random
from concurrent.futures import ThreadPoolExecutor

from registry.prompt import *
from utils.guidline_utils import *


def introduces_blank(orig_sentence, transformed_sentence):
    return '<blank>' not in orig_sentence and '<blank>' in transformed_sentence


def _rule_budget_reached(applied_rules, max_rules):
    return max_rules is not None and len(applied_rules) >= max_rules


def _rule_usage_cap_reached(feature, rule_usage_counts, max_rule_applications_per_rule):
    if rule_usage_counts is None or max_rule_applications_per_rule is None:
        return False
    return rule_usage_counts.get(feature, 0) >= max_rule_applications_per_rule


def _order_guidelines(guideline, rule_usage_counts, rule_balance_strength=0.0):
    ordered = list(guideline)
    random.shuffle(ordered)

    if rule_usage_counts is None or rule_balance_strength is None or rule_balance_strength <= 0:
        return ordered

    return sorted(
        ordered,
        key=lambda item: (
            rule_usage_counts.get(item[0], 0) * rule_balance_strength,
            random.random(),
        ),
    )


def _run_in_parallel(items, max_workers, func):
    if max_workers is None or max_workers <= 1 or len(items) <= 1:
        return [func(item) for item in items]

    with ThreadPoolExecutor(max_workers=min(max_workers, len(items))) as executor:
        return list(executor.map(func, items))


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



def transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config, max_rules_per_chunk=None, rule_usage_counts=None, max_rule_applications_per_rule=None, rule_balance_strength=0.0):
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
    guideline = _order_guidelines(guideline, rule_usage_counts, rule_balance_strength)

    for i in range(len(guideline)):
        feature = guideline[i][0]
        if _rule_usage_cap_reached(feature, rule_usage_counts, max_rule_applications_per_rule):
            continue

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
                    if _rule_usage_cap_reached(feature, rule_usage_counts, max_rule_applications_per_rule):
                        continue
                    sentence[num] = transformed_sentence
                    applied_rules[num].append(feature)
                    if rule_usage_counts is not None:
                        rule_usage_counts[feature] += 1
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



def openai_transformation(sentence, guideline, client, sampling_params, task_config, model_config, max_rules_per_chunk=None, rule_usage_counts=None, max_rule_applications_per_rule=None, rule_balance_strength=0.0, openai_parallelism=1):
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

    guideline = _order_guidelines(guideline, rule_usage_counts, rule_balance_strength)

    transformation_params = {
        'temperature': sampling_params['temperature'],
        'top_p': sampling_params['top_p'],
        'max_tokens': sampling_params['max_tokens'],
    }
    semantic_model_name = model_config.semantic_model_name or model_config.model_name
    consecutive_model_errors = 0
    max_consecutive_model_errors = 10

    for i in range(len(guideline)):
        feature = guideline[i][0]
        if _rule_usage_cap_reached(feature, rule_usage_counts, max_rule_applications_per_rule):
            continue

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

        candidate_tasks = []
        for num, response, error in _run_in_parallel(active_prompts, openai_parallelism, call_transformation):
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
                if _rule_usage_cap_reached(feature, rule_usage_counts, max_rule_applications_per_rule):
                    continue
                sentence[num] = transformed_sentence
                applied_rules[num].append(feature)
                if rule_usage_counts is not None:
                    rule_usage_counts[feature] += 1
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
