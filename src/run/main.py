import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import re
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from collections import defaultdict
from transformers import AutoTokenizer

from torch.utils.data import DataLoader

from configs.parse_arguments import parse_args
from framework.guideline import return_guideline
from framework.data_return import return_dataloader
from framework.transformation import transformation, openai_transformation
from registry.framework import QUESTION_KEY_ID
from utils import log, colorstr
from utils.common import save_func
from utils.cefr_texts import (
    INTERNAL_EMPTY_TEXT_COLUMN,
    empty_transformation_result,
    load_cefr_text_dataset,
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



def main():
    # Initialize arguments
    generation_config, model_config, dataset_config, task_config, save_config = parse_args()

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
        tokenizer_name = model_config.tokenizer or model_config.model_name
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, cache_dir=os.environ.get("MODEL_DIR", None))

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
        dataloader = DataLoader(dataset, generation_config.batch_size, shuffle=False)

    elif task_config.task_name == 'cefr':
        dataset = load_dataset(
            "csv",
            data_files = {"test": f'{save_config.data_path}/assets/vocab_processed/{dataset_config.dataset_name}_{(task_config.cefr_level).lower()}.csv'},
            split="test",
        )

        if generation_config.rerun is not None:
            rerun_index = list(np.load(generation_config.rerun))
            dataset = dataset.select(rerun_index)

        dataloader = DataLoader(dataset, generation_config.batch_size, shuffle=False)


    elif task_config.task_name == 'L1':
        cefr_data_path = ('/').join(save_config.save_path.split('/')[:-2])

        dataset = load_dataset(
            "csv",  
            data_files={"test": f'{cefr_data_path}/assets/cefr/{dataset_config.dataset_name}/{task_config.cefr_level}.csv'},
            split='test',
        )

        if generation_config.rerun is not None:
            rerun_index = list(np.load(generation_config.rerun))
            dataset = dataset.select(rerun_index)
    
        dataloader = DataLoader(dataset, generation_config.batch_size, shuffle=False)

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

            transform_indices = [idx for idx, is_empty in enumerate(empty_mask) if not is_empty]
            transformed_batch = [None for _ in range(len(sentence))]

            if transform_indices:
                transform_sentence = [sentence[idx] for idx in transform_indices]

                if use_hosted_openai:
                    transformed_results = openai_transformation(transform_sentence, guideline, client, sampling_params, task_config, model_config)
                else:
                    transformed_results = transformation(transform_sentence, guideline, client, tokenizer, sampling_params, task_config, model_config)

                for idx, output in zip(transform_indices, transformed_results):
                    transformed_batch[idx] = output

            for idx, value in enumerate(sentence):
                if transformed_batch[idx] is None:
                    transformed_batch[idx] = empty_transformation_result(value)

            iter_result = transformed_batch
        else:
            sentence = [re.sub(r'_{2,}', '<blank>', s) for s in sentence]

            if use_hosted_openai:
                iter_result = openai_transformation(sentence, guideline, client, sampling_params, task_config, model_config)
            else:
                iter_result = transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config)

        to_save.extend(iter_result)

        if dataset_config.dataset_name in choice_transform_dataset:

            # choices transform
            for choice_num, sentence in enumerate(sample['choices']['text']):
                iter_result = transformation(sentence, guideline, client, tokenizer, sampling_params, task_config, model_config)
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
