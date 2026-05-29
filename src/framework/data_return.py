import os
import json
import numpy as np
import pandas as pd
from collections import Counter

from registry.dataset_map import  DATASET_MAPPING
from utils.cefr_texts import DIAGNOSTIC_COUNT_KEYS, load_cefr_text_frame
from utils.cefr_filter import write_caa_pair_files, parse_rule_list
from utils import log



def return_dataloader(dataset_config, generation_config, start_idx=None):
    rerun_index = None
    if generation_config.rerun is not None:
        rerun_index = list(np.load(generation_config.rerun))
    return DATASET_MAPPING[dataset_config.dataset_name](generation_config.batch_size, rerun_index, start_idx)


def _changed_chunk_records(output):
    records = output.get('chunks', [])
    if not records:
        return []

    changed_records = []
    for record in records:
        rules = record.get('applied_rules', []) or []
        is_changed = record.get('is_changed', False)
        if rules or is_changed:
            changed_records.append({
                'chunk_index': record.get('chunk_index'),
                'orig_text': record.get('orig_text', ''),
                'transformed_text': record.get('transformed_text', record.get('orig_text', '')),
                'applied_rules': rules,
            })

    return changed_records


def _truncate_text(value, max_chars=500):
    if value is None:
        return ""
    value = str(value).replace("\n", "\\n")
    if len(value) <= max_chars:
        return value
    return value[:max_chars - 3] + "..."


def _first_nonempty(values):
    for value in values:
        if value:
            return value
    return ""


def _config_json(config):
    if config is None:
        return "{}"
    if hasattr(config, "__dataclass_fields__"):
        values = {key: getattr(config, key) for key in config.__dataclass_fields__}
    else:
        values = dict(vars(config))
    return json.dumps(values, sort_keys=True, ensure_ascii=False)


def _rule_distribution(values):
    counts = Counter()
    for value in values:
        counts.update(parse_rule_list(value))
    return counts


def return_cefr_texts(to_save, save_config, dataset_config, generation_config, task_config=None, model_config=None):
    rerun_index = None
    if generation_config.rerun is not None:
        rerun_index = list(np.load(generation_config.rerun))

    source_df = load_cefr_text_frame(dataset_config, rerun_index)
    outputs = to_save.get('question', [])
    source_df = source_df.iloc[:len(outputs)].copy()
    if '__transenv_row_idx' in source_df.columns:
        source_df['source_row_idx'] = source_df['__transenv_row_idx']
    else:
        source_df['source_row_idx'] = source_df.index

    orig_sentences = []
    transformed_texts = []
    applied_rules = []
    num_applied_rules = []
    is_changed = []
    chunk_counts = []
    changed_chunk_counts = []
    changed_chunk_indices = []
    changed_chunks = []
    model_response_counts = []
    candidate_transform_counts = []
    semantic_judge_counts = []
    diagnostic_columns = {key: [] for key in DIAGNOSTIC_COUNT_KEYS}
    sample_model_errors = []
    sample_semantic_errors = []
    sample_model_responses = []

    for output in outputs:
        orig_sentence = output.get('orig_sentence', '')
        final_sentence = output.get('final_sentence', orig_sentence)
        rules = output.get('applied_rules', output.get('applied_rule', []))
        changed_chunk_records = _changed_chunk_records(output)

        orig_sentences.append(orig_sentence)
        transformed_texts.append(final_sentence)
        applied_rules.append(json.dumps(rules, ensure_ascii=False))
        num_applied_rules.append(len(rules))
        is_changed.append(final_sentence != orig_sentence)
        chunk_counts.append(output.get('chunk_count', 1))
        changed_chunk_counts.append(len(changed_chunk_records))
        changed_chunk_indices.append(json.dumps([record['chunk_index'] for record in changed_chunk_records], ensure_ascii=False))
        changed_chunks.append(json.dumps(changed_chunk_records, ensure_ascii=False))
        model_response_counts.append(len(output.get('whole_response', [])))
        candidate_transform_counts.append(len(output.get('mid_transformed_sentences', [])))
        semantic_judge_counts.append(len(output.get('judge_repsonse', [])))
        for key in DIAGNOSTIC_COUNT_KEYS:
            diagnostic_columns[key].append(output.get(key, 0))
        sample_model_errors.append(_truncate_text(_first_nonempty(output.get('model_errors', []))))
        sample_semantic_errors.append(_truncate_text(_first_nonempty(output.get('semantic_errors', []))))
        sample_model_responses.append(_truncate_text(_first_nonempty(output.get('whole_response', []))))

    source_df['orig_sentence'] = orig_sentences
    source_df['transformed_text'] = transformed_texts
    source_df['applied_rules'] = applied_rules
    source_df['num_applied_rules'] = num_applied_rules
    source_df['is_changed'] = is_changed
    source_df['chunk_count'] = chunk_counts
    source_df['changed_chunk_count'] = changed_chunk_counts
    source_df['changed_chunk_indices'] = changed_chunk_indices
    source_df['changed_chunks'] = changed_chunks
    source_df['model_response_count'] = model_response_counts
    source_df['candidate_transform_count'] = candidate_transform_counts
    source_df['semantic_judge_count'] = semantic_judge_counts
    for key, values in diagnostic_columns.items():
        source_df[key] = values
    source_df['sample_model_error'] = sample_model_errors
    source_df['sample_semantic_error'] = sample_semantic_errors
    source_df['sample_model_response'] = sample_model_responses

    if '__transenv_row_idx' in source_df.columns:
        source_df = source_df.drop(columns=['__transenv_row_idx'])

    suffix = '_rerun' if generation_config.rerun is not None else ''
    raw_path = os.path.join(save_config.save_path, f'{save_config.file_name}{suffix}.csv')
    source_df.to_csv(raw_path, index=False)

    if getattr(generation_config, "write_caa_pairs", False):
        dialect = getattr(task_config, "dialect", "") if task_config is not None else ""
        transform_model = getattr(model_config, "model_name", "") if model_config is not None else ""
        semantic_model = ""
        if model_config is not None:
            semantic_model = getattr(model_config, "semantic_model_name", None) or transform_model
        result = write_caa_pair_files(
            source_df,
            output_dir=save_config.save_path,
            file_prefix=f'{save_config.file_name}{suffix}',
            dialect=dialect,
            transform_model=transform_model,
            semantic_model=semantic_model,
            generation_config_json=_config_json(generation_config),
            max_edit_rate=getattr(generation_config, "caa_max_edit_rate", 0.50),
            min_length_ratio=getattr(generation_config, "caa_min_length_ratio", 0.50),
            max_length_ratio=getattr(generation_config, "caa_max_length_ratio", 1.80),
        )
        rule_counts = _rule_distribution(source_df[source_df['is_changed']]['applied_rules'])
        top_rules = ", ".join(f"{rule}:{count}" for rule, count in rule_counts.most_common(8)) or "<none>"
        log(
            "CAA pair export: "
            f"raw_rows={len(source_df)}, accepted_pairs={result['kept_rows']}, "
            f"dropped_rows={result['dropped_rows']}, top_rules={top_rules}, "
            f"pairs_path={result['pairs_path']}"
        )



def return_mmlu(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:

        columns = list(test_dataset.features.keys())

        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'question': output['final_sentence'].replace('<blank>', '_______'),
                    'subject': test_dataset[i]['subject'],
                    'choices': [test_dataset[i]['choices']],
                    'answer': test_dataset[i]['answer']
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'question': output['final_sentence'].replace('<blank>', '_______'),
                    'subject': test_dataset[cefr_index[i]]['subject'],
                    'choices': [test_dataset[cefr_index[i]]['choices']],
                    'answer': test_dataset[cefr_index[i]]['answer']
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'), index=False)
        
    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv')) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'))
        

        for i, output in enumerate(to_save['question']):
            index = int(rerun_index[i])
            new_row = {
                'question': output['final_sentence'].replace('<blank>', '_______'),
                'subject': test_dataset[index]['subject'],
                'choices': [test_dataset[index]['choices']],
                'answer': test_dataset[index]['answer']
            }

            df.loc[index] = new_row
        
        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'), index=False)



def return_gsm8k(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'question': output['final_sentence'],
                    'answer': test_dataset[i]['answer'],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'question': output['final_sentence'],
                    'answer': test_dataset[cefr_index[i]]['answer'],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'), index=False)
    
    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv')) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'))

        for i, output in enumerate(to_save['question']):
            index = int(rerun_index[i])
            new_row = {
                'question': output['final_sentence'],
                'answer': test_dataset[index]['answer'],
            }
        
            df.loc[index] = new_row
        
        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'), index=False)



def return_arc(test_dataset, to_save_dict, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())

        df= pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save_dict['question']):
                new_row = pd.DataFrame({
                    'id': test_dataset[i]['id'],
                    'question': output['final_sentence'],
                    'choices': [test_dataset[i]['choices']],
                    'answerKey': test_dataset[i]['answerKey']
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)
        
        elif cefr_index is not None:
            for i, output in enumerate(to_save_dict['question']):
                new_row = pd.DataFrame({
                    'id': test_dataset[cefr_index[i]]['id'],
                    'question': output['final_sentence'],
                    'choices': [test_dataset[cefr_index[i]]['choices']],
                    'answerKey': test_dataset[cefr_index[i]]['answerKey']
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'), index=False)
    
    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv')) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'))

        for i, output in enumerate(to_save_dict['question']):
            index = int(rerun_index[i])
            new_row = {
                'id': test_dataset[index]['id'],
                'question': output['final_sentence'],
                'choices': [test_dataset[index]['choices']],
                'answerKey': test_dataset[index]['answerKey']
            }

            df.loc[index] = new_row
        
        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'), index=False)



def return_hellaswag(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'ind': test_dataset[i]['ind'],
                    'activity_label': test_dataset[i]['activity_label'],
                    'ctx_a': output['final_sentence'],
                    'ctx_b': test_dataset[i]['ctx_b'],
                    'ctx': output['final_sentence'] + ' ' + test_dataset[i]['ctx_b'],
                    'endings': [test_dataset[i]['endings']],
                    'source_id': test_dataset[i]['source_id'],
                    'split': test_dataset[i]['split'],
                    'split_type': test_dataset[i]['split_type'],
                    'label': test_dataset[i]['label'],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)
        
        elif cefr_index is not None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'ind': test_dataset[cefr_index[i]]['ind'],
                    'activity_label': test_dataset[cefr_index[i]]['activity_label'],
                    'ctx_a': test_dataset[cefr_index[i]]['ctx_a'],
                    'ctx_b': test_dataset[cefr_index[i]]['ctx_b'],
                    'ctx': output['final_sentence'],
                    'endings': [test_dataset[cefr_index[i]]['endings']],
                    'source_id': test_dataset[cefr_index[i]]['source_id'],
                    'split': test_dataset[cefr_index[i]]['split'],
                    'split_type': test_dataset[cefr_index[i]]['split_type'],
                    'label': test_dataset[cefr_index[i]]['label'],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'), index=False)
    
    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv')) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'))

        for i, output in enumerate(to_save['question']):
            index = int(rerun_index[i])
            new_row = {
                'ind': test_dataset[index]['ind'],
                'activity_label': test_dataset[index]['activity_label'],
                'ctx_a': output['final_sentence'],
                'ctx_b': test_dataset[index]['ctx_b'],
                'ctx': output['final_sentence'],
                'endings': test_dataset[index]['endings'],
                'source_id': test_dataset[index]['source_id'],
                'split': test_dataset[index]['split'],
                'split_type': test_dataset[index]['split_type'],
                'label': test_dataset[index]['label'],
            }
        
            df.loc[index] = new_row
        
        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'), index=False)
    
        

def return_truthfulqa(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'question': output['final_sentence'],
                    'mc1_targets': [test_dataset[i]['mc1_targets']],
                    'mc2_targets': [test_dataset[i]['mc2_targets']],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'question': output['final_sentence'],
                    'mc1_targets': [test_dataset[cefr_index[i]]['mc1_targets']],
                    'mc2_targets': [test_dataset[cefr_index[i]]['mc2_targets']],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)


        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'), index=False)
    
    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv')) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'))

        for i, output in enumerate(to_save['question']):
            index = int(rerun_index[i])
            new_row = {
                'question': output['final_sentence'],
                'mc1_targets': test_dataset[index]['mc1_targets'],
                'mc2_targets': test_dataset[index]['mc2_targets']
            }
        
            df.loc[index] = new_row
        
        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'), index=False)



def return_winogrande(test_dataset, to_save, save_config, rerun_index=None, cefr_index=None):
    if rerun_index is None:
        columns = list(test_dataset.features.keys())
        df = pd.DataFrame(columns=columns)

        if cefr_index is None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'sentence': output['final_sentence'],
                    'option1': test_dataset[i]['option1'],
                    'option2': test_dataset[i]['option2'],
                    'answer': test_dataset[i]['answer'],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        elif cefr_index is not None:
            for i, output in enumerate(to_save['question']):
                new_row = pd.DataFrame({
                    'sentence': output['final_sentence'],
                    'option1': test_dataset[cefr_index[i]]['option1'],
                    'option2': test_dataset[cefr_index[i]]['option2'],
                    'answer': test_dataset[cefr_index[i]]['answer'],
                }, index=[0])

                df = pd.concat([df, new_row], ignore_index=True)

        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'), index=False)
    
    elif rerun_index is not None:
        if os.path.isfile(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv')) is True:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'))
        else:
            df = pd.read_csv(os.path.join(save_config.save_path, f'{save_config.file_name}.csv'))

        for i, output in enumerate(to_save['question']):
            index = int(rerun_index[i])
            new_row = {
                'sentence': output['final_sentence'],
                'option1': test_dataset[index]['option1'],
                'option2': test_dataset[index]['option2'],
                'answer': test_dataset[index]['answer'],
            }

            df.loc[index] = new_row
        
        df.to_csv(os.path.join(save_config.save_path, f'{save_config.file_name}_rerun.csv'), index=False)
