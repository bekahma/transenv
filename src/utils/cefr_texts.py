import os

import pandas as pd


INTERNAL_TEXT_COLUMN = "__transenv_text"
INTERNAL_ROW_INDEX_COLUMN = "__transenv_row_idx"
INTERNAL_EMPTY_TEXT_COLUMN = "__transenv_empty_text"

TEXT_COLUMN_CANDIDATES = ("text", "story", "description", "sentence", "content")
LEVEL_COLUMN_CANDIDATES = ("level", "cefr_level", "cefr", "label")
CEFR_LEVEL_GROUPS = {
    "A": ("A1", "A2"),
    "B": ("B1", "B2"),
    "C": ("C1", "C2"),
}


def _normalize_name(value):
    return str(value).strip().lower()


def _normalize_text(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def parse_cefr_levels(levels):
    if levels is None:
        return None

    parsed = []
    for level in str(levels).split(","):
        level = level.strip().upper()
        if not level:
            continue
        parsed.extend(CEFR_LEVEL_GROUPS.get(level, (level,)))

    return set(parsed) if parsed else None


def detect_text_column(column_names, text_column=None):
    columns = list(column_names)
    if text_column is not None:
        if text_column not in columns:
            raise ValueError(f"Text column '{text_column}' was not found. Available columns: {columns}")
        return text_column

    normalized = {_normalize_name(column): column for column in columns}
    for candidate in TEXT_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]

    if len(columns) == 1:
        return columns[0]

    raise ValueError(
        "Could not infer the text column for cefr_texts. "
        "Pass --text_column explicitly. "
        f"Available columns: {columns}"
    )


def detect_level_column(column_names):
    normalized = {_normalize_name(column): column for column in column_names}
    for candidate in LEVEL_COLUMN_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    return None


def validate_cefr_text_config(dataset_config):
    if dataset_config.input_path is None:
        raise ValueError("--input_path is required when --dataset_name cefr_texts")
    if not os.path.exists(dataset_config.input_path):
        raise FileNotFoundError(f"Could not find cefr_texts input CSV: {dataset_config.input_path}")


def _filter_dataset_by_levels(dataset, dataset_config):
    levels = parse_cefr_levels(dataset_config.input_cefr_levels)
    if levels is None:
        return dataset

    level_column = detect_level_column(dataset.column_names)
    if level_column is None:
        raise ValueError(
            "--input_cefr_levels was provided, but no CEFR level column was found. "
            f"Expected one of {list(LEVEL_COLUMN_CANDIDATES)}; available columns: {dataset.column_names}"
        )

    return dataset.filter(
        lambda row: str(row[level_column]).strip().upper() in levels,
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def _filter_frame_by_levels(df, dataset_config):
    levels = parse_cefr_levels(dataset_config.input_cefr_levels)
    if levels is None:
        return df

    level_column = detect_level_column(df.columns)
    if level_column is None:
        raise ValueError(
            "--input_cefr_levels was provided, but no CEFR level column was found. "
            f"Expected one of {list(LEVEL_COLUMN_CANDIDATES)}; available columns: {list(df.columns)}"
        )

    return df[df[level_column].astype(str).str.strip().str.upper().isin(levels)]


def load_cefr_text_dataset(dataset_config, generation_config, start_idx=0):
    validate_cefr_text_config(dataset_config)

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "cefr_texts loading requires the Hugging Face datasets package. "
            "Install project requirements with `pip install -r requirements.txt`."
        ) from exc

    dataset = load_dataset(
        "csv",
        data_files={"test": dataset_config.input_path},
        split="test",
        keep_in_memory=True,
    )

    text_column = detect_text_column(dataset.column_names, dataset_config.text_column)

    dataset = dataset.map(
        lambda row, idx: {INTERNAL_ROW_INDEX_COLUMN: idx},
        with_indices=True,
        load_from_cache_file=False,
        keep_in_memory=True,
    )
    dataset = _filter_dataset_by_levels(dataset, dataset_config)

    if generation_config.rerun is not None:
        import numpy as np

        dataset = dataset.select(list(np.load(generation_config.rerun)))

    max_samples = getattr(generation_config, "max_samples", None)
    if max_samples is not None:
        if max_samples < 0:
            raise ValueError("--max_samples must be greater than or equal to 0")
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    if start_idx:
        dataset = dataset.skip(start_idx)

    def add_internal_text(row):
        text = _normalize_text(row[text_column])
        return {
            INTERNAL_TEXT_COLUMN: text,
            INTERNAL_EMPTY_TEXT_COLUMN: len(text.strip()) == 0,
        }

    return dataset.map(
        add_internal_text,
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def load_cefr_text_frame(dataset_config, rerun_index=None):
    validate_cefr_text_config(dataset_config)

    df = pd.read_csv(dataset_config.input_path)
    detect_text_column(df.columns, dataset_config.text_column)
    df[INTERNAL_ROW_INDEX_COLUMN] = range(len(df))
    df = _filter_frame_by_levels(df, dataset_config)

    if rerun_index is not None:
        df = df.iloc[[int(index) for index in rerun_index]]

    return df.reset_index(drop=True)


def empty_transformation_result(sentence):
    sentence = _normalize_text(sentence)
    return {
        "orig_sentence": sentence,
        "whole_response": [],
        "mid_transformed_sentences": [],
        "judge_repsonse": [],
        "applied_rules": [],
        "transformed_sentences": [],
        "final_sentence": sentence,
    }
