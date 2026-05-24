import os


def _load_dataset(*args, **kwargs):
    from datasets import load_dataset

    return load_dataset(*args, **kwargs)


def _mmlu_load_test_data(args):
    from benchmark.mmlu import load_test_data

    return load_test_data(args)


def _mmlu_extract_answer(outputs):
    from benchmark.mmlu import extract_answer

    return extract_answer(outputs)


def _gsm8k_load_test_data(args):
    from benchmark.gsm8k import load_test_data

    return load_test_data(args)


def _gsm8k_extract_answer(outputs):
    from benchmark.gsm8k import extract_answer

    return extract_answer(outputs)


def _arc_load_test_data(args):
    from benchmark.arc import load_test_data

    return load_test_data(args)


def _arc_extract_answer(outputs):
    from benchmark.arc import extract_answer

    return extract_answer(outputs)


def _hellaswag_load_test_data(args):
    from benchmark.hellaswag import load_test_data

    return load_test_data(args)


def _hellaswag_extract_answer(outputs):
    from benchmark.hellaswag import extract_answer

    return extract_answer(outputs)


def _truthfulqa_load_test_data(args):
    from benchmark.truthful_qa import load_test_data

    return load_test_data(args)


def _truthfulqa_extract_answer(outputs):
    from benchmark.truthful_qa import extract_answer

    return extract_answer(outputs)


def _winogrande_load_test_data(args):
    from benchmark.winogrande import load_test_data

    return load_test_data(args)


def _winogrande_extract_answer(outputs):
    from benchmark.winogrande import extract_answer

    return extract_answer(outputs)


def _mmlu_dataloader(*args, **kwargs):
    from benchmark.mmlu import mmlu_dataloader

    return mmlu_dataloader(*args, **kwargs)


def _gsm8k_dataloader(*args, **kwargs):
    from benchmark.gsm8k import gsm8k_dataloader

    return gsm8k_dataloader(*args, **kwargs)


def _arc_dataloader(*args, **kwargs):
    from benchmark.arc import arc_dataloader

    return arc_dataloader(*args, **kwargs)


def _hellaswag_dataloader(*args, **kwargs):
    from benchmark.hellaswag import hellaswag_dataloader

    return hellaswag_dataloader(*args, **kwargs)


def _truthfulqa_dataloader(*args, **kwargs):
    from benchmark.truthful_qa import truthfulqa_dataloader

    return truthfulqa_dataloader(*args, **kwargs)


def _winogrande_dataloader(*args, **kwargs):
    from benchmark.winogrande import winogrande_dataloader

    return winogrande_dataloader(*args, **kwargs)



# Function definition of each benchmark
MAIN_FUNCS = {
    "mmlu": (_mmlu_load_test_data, _mmlu_extract_answer, 13436, "question"),
    "gsm8k": (_gsm8k_load_test_data, _gsm8k_extract_answer, 1319, "question"),
    "arc": (_arc_load_test_data, _arc_extract_answer, 1172, "question"),
    "hellaswag": (_hellaswag_load_test_data, _hellaswag_extract_answer, 10042, "ctx"),
    "truthful_qa": (
        _truthfulqa_load_test_data,
        _truthfulqa_extract_answer,
        817,
        "question",
    ),
    "winogrande": (
        _winogrande_load_test_data,
        _winogrande_extract_answer,
        1267,
        "sentence",
    ),
}



def load_test_dataset(dataset_name):
    if dataset_name == 'mmlu':
        from benchmark.mmlu import load_mmlu_test_dataset

        return load_mmlu_test_dataset()
    if dataset_name == 'gsm8k':
        return _load_dataset('openai/gsm8k', 'main', split='test', cache_dir=os.environ.get("DATA_DIR", None))
    if dataset_name == 'arc':
        return _load_dataset('allenai/ai2_arc', 'ARC-Challenge', split='test', cache_dir=os.environ.get("DATA_DIR", None))
    if dataset_name == 'hellaswag':
        return _load_dataset('Rowan/hellaswag', split='validation', cache_dir=os.environ.get("DATA_DIR", None))
    if dataset_name == 'truthfulqa':
        return _load_dataset('truthfulqa/truthful_qa', 'multiple_choice', split='validation', cache_dir=os.environ.get("DATA_DIR", None))
    if dataset_name == 'winogrande':
        return _load_dataset('allenai/winogrande', split='validation', trust_remote_code=True, name='winogrande_m', cache_dir=os.environ.get("DATA_DIR", None))
    raise KeyError(f"Unknown benchmark dataset: {dataset_name}")



# Dataloaders
DATASET_MAPPING = {
    'mmlu': _mmlu_dataloader,
    'gsm8k': _gsm8k_dataloader,
    'arc': _arc_dataloader,
    'hellaswag': _hellaswag_dataloader,
    'truthfulqa': _truthfulqa_dataloader,
    'winogrande': _winogrande_dataloader,
}
