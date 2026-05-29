import json
import os
import re
from difflib import SequenceMatcher

import pandas as pd


WORD_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?")
CAA_PAIR_COLUMNS = [
    "pair_id",
    "source_row_idx",
    "cefr_level",
    "dialect",
    "sae_text",
    "aae_text",
    "applied_rules",
    "num_applied_rules",
    "source_word_count",
    "transformed_word_count",
    "qa_word_edit_rate",
    "qa_length_ratio",
    "model_response_count",
    "semantic_judge_count",
    "transform_model",
    "semantic_model",
    "generation_config_json",
]


def _as_text(value):
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value)


def _as_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return default
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return default


def _as_int(value, default=0):
    if value is None:
        return default
    if pd.isna(value):
        return default
    try:
        return int(value)
    except Exception:
        return default


def parse_rule_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []

    try:
        parsed = json.loads(str(value))
    except Exception:
        return []

    return parsed if isinstance(parsed, list) else []


def word_tokens(text):
    return WORD_PATTERN.findall(_as_text(text))


def word_edit_stats(orig_text, transformed_text):
    orig_words = word_tokens(orig_text)
    transformed_words = word_tokens(transformed_text)
    matcher = SequenceMatcher(a=orig_words, b=transformed_words, autojunk=False)

    insertions = 0
    deletions = 0
    replacements = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            insertions += j2 - j1
        elif tag == "delete":
            deletions += i2 - i1
        elif tag == "replace":
            replacements += max(i2 - i1, j2 - j1)

    edits = insertions + deletions + replacements
    edit_rate = edits / len(orig_words) if orig_words else 0

    return {
        "qa_orig_word_count": len(orig_words),
        "qa_transformed_word_count": len(transformed_words),
        "qa_word_edit_count": edits,
        "qa_word_edit_rate": edit_rate,
        "qa_word_insertions": insertions,
        "qa_word_deletions": deletions,
        "qa_word_replacements": replacements,
    }


def adjacent_duplicate_tokens(text):
    tokens = [token.lower() for token in word_tokens(text)]
    duplicates = []

    for idx in range(1, len(tokens)):
        if tokens[idx] == tokens[idx - 1] and len(tokens[idx]) > 1:
            duplicates.append(tokens[idx])

    return duplicates


def new_adjacent_duplicate_tokens(orig_text, transformed_text):
    orig_duplicates = set(adjacent_duplicate_tokens(orig_text))
    transformed_duplicates = adjacent_duplicate_tokens(transformed_text)
    return sorted({token for token in transformed_duplicates if token not in orig_duplicates})


def evaluate_cefr_filter_row(
    row,
    text_column="orig_sentence",
    transformed_column="transformed_text",
    applied_rules_column="applied_rules",
    changed_column="is_changed",
    num_rules_column="num_applied_rules",
    max_edit_rate=0.10,
    min_length_ratio=0.50,
    max_length_ratio=1.50,
    keep_unchanged=False,
    allow_new_adjacent_duplicates=False,
    model_error_count_column="model_error_count",
    semantic_error_count_column="semantic_error_count",
    drop_model_errors=True,
    drop_semantic_errors=True,
):
    orig_text = _as_text(row.get(text_column, ""))
    transformed_text = _as_text(row.get(transformed_column, ""))
    rules = parse_rule_list(row.get(applied_rules_column, "[]"))

    if changed_column in row:
        is_changed = _as_bool(row.get(changed_column), default=transformed_text != orig_text)
    else:
        is_changed = transformed_text != orig_text

    if num_rules_column in row and not pd.isna(row.get(num_rules_column)):
        num_rules = int(row.get(num_rules_column))
    else:
        num_rules = len(rules)

    stats = word_edit_stats(orig_text, transformed_text)
    length_ratio = len(transformed_text) / len(orig_text) if orig_text else 0
    new_duplicates = new_adjacent_duplicate_tokens(orig_text, transformed_text)

    reasons = []
    if not transformed_text.strip():
        reasons.append("empty_transformed_text")
    if "<blank>" in transformed_text and "<blank>" not in orig_text:
        reasons.append("introduced_blank")
    if not keep_unchanged and (not is_changed or num_rules == 0):
        reasons.append("unchanged_or_no_applied_rules")
    if drop_model_errors and _as_int(row.get(model_error_count_column, 0)) > 0:
        reasons.append("model_errors")
    if drop_semantic_errors and _as_int(row.get(semantic_error_count_column, 0)) > 0:
        reasons.append("semantic_errors")
    if max_edit_rate is not None and stats["qa_word_edit_rate"] > max_edit_rate:
        reasons.append(f"edit_rate_gt_{max_edit_rate:.2f}")
    if min_length_ratio is not None and orig_text and length_ratio < min_length_ratio:
        reasons.append(f"length_ratio_lt_{min_length_ratio:.2f}")
    if max_length_ratio is not None and orig_text and length_ratio > max_length_ratio:
        reasons.append(f"length_ratio_gt_{max_length_ratio:.2f}")
    if not allow_new_adjacent_duplicates and new_duplicates:
        reasons.append("new_adjacent_duplicate_tokens")

    return {
        **stats,
        "qa_length_ratio": length_ratio,
        "qa_new_adjacent_duplicates": json.dumps(new_duplicates, ensure_ascii=False),
        "filter_keep": len(reasons) == 0,
        "filter_reasons": json.dumps(reasons, ensure_ascii=False),
        "transformed_text_filtered": transformed_text if len(reasons) == 0 else "",
    }


def filter_cefr_dataframe(
    df,
    text_column="orig_sentence",
    transformed_column="transformed_text",
    applied_rules_column="applied_rules",
    changed_column="is_changed",
    num_rules_column="num_applied_rules",
    max_edit_rate=0.10,
    min_length_ratio=0.50,
    max_length_ratio=1.50,
    keep_unchanged=False,
    allow_new_adjacent_duplicates=False,
    model_error_count_column="model_error_count",
    semantic_error_count_column="semantic_error_count",
    drop_model_errors=True,
    drop_semantic_errors=True,
):
    required_columns = [text_column, transformed_column]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    audit_rows = []
    for _, row in df.iterrows():
        audit_rows.append(
            evaluate_cefr_filter_row(
                row,
                text_column=text_column,
                transformed_column=transformed_column,
                applied_rules_column=applied_rules_column,
                changed_column=changed_column,
                num_rules_column=num_rules_column,
                max_edit_rate=max_edit_rate,
                min_length_ratio=min_length_ratio,
                max_length_ratio=max_length_ratio,
                keep_unchanged=keep_unchanged,
                allow_new_adjacent_duplicates=allow_new_adjacent_duplicates,
                model_error_count_column=model_error_count_column,
                semantic_error_count_column=semantic_error_count_column,
                drop_model_errors=drop_model_errors,
                drop_semantic_errors=drop_semantic_errors,
            )
        )

    audit_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(audit_rows)], axis=1)
    filtered_df = audit_df[audit_df["filter_keep"]].copy()
    dropped_df = audit_df[~audit_df["filter_keep"]].copy()

    return audit_df, filtered_df, dropped_df


def _first_present(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def _slug(value):
    value = _as_text(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "dialect"


def build_caa_pairs_dataframe(
    filtered_df,
    dialect="",
    transform_model="",
    semantic_model="",
    generation_config_json="",
    text_column="orig_sentence",
    transformed_column="transformed_text",
    applied_rules_column="applied_rules",
    num_rules_column="num_applied_rules",
):
    level_column = _first_present(filtered_df.columns, ("label", "level", "cefr_level", "cefr"))
    source_row_column = _first_present(filtered_df.columns, ("source_row_idx", "__transenv_row_idx"))
    dialect_slug = _slug(dialect)

    rows = []
    for idx, row in filtered_df.reset_index(drop=True).iterrows():
        source_row_idx = row.get(source_row_column, idx) if source_row_column else idx
        rules = parse_rule_list(row.get(applied_rules_column, "[]"))
        rows.append({
            "pair_id": f"{dialect_slug}:{source_row_idx}",
            "source_row_idx": source_row_idx,
            "cefr_level": row.get(level_column, "") if level_column else "",
            "dialect": dialect,
            "sae_text": _as_text(row.get(text_column, "")),
            "aae_text": _as_text(row.get(transformed_column, "")),
            "applied_rules": json.dumps(rules, ensure_ascii=False),
            "num_applied_rules": _as_int(row.get(num_rules_column, len(rules)), len(rules)),
            "source_word_count": _as_int(row.get("qa_orig_word_count", len(word_tokens(row.get(text_column, ""))))),
            "transformed_word_count": _as_int(row.get("qa_transformed_word_count", len(word_tokens(row.get(transformed_column, ""))))),
            "qa_word_edit_rate": row.get("qa_word_edit_rate", 0),
            "qa_length_ratio": row.get("qa_length_ratio", 0),
            "model_response_count": _as_int(row.get("model_response_count", 0)),
            "semantic_judge_count": _as_int(row.get("semantic_judge_count", 0)),
            "transform_model": transform_model,
            "semantic_model": semantic_model,
            "generation_config_json": generation_config_json,
        })

    return pd.DataFrame(rows, columns=CAA_PAIR_COLUMNS)


def write_caa_pair_files(
    df,
    output_dir,
    file_prefix,
    dialect="",
    transform_model="",
    semantic_model="",
    generation_config_json="",
    text_column="orig_sentence",
    transformed_column="transformed_text",
    applied_rules_column="applied_rules",
    changed_column="is_changed",
    num_rules_column="num_applied_rules",
    max_edit_rate=0.50,
    min_length_ratio=0.50,
    max_length_ratio=1.80,
):
    os.makedirs(output_dir, exist_ok=True)

    audit_df, filtered_df, dropped_df = filter_cefr_dataframe(
        df,
        text_column=text_column,
        transformed_column=transformed_column,
        applied_rules_column=applied_rules_column,
        changed_column=changed_column,
        num_rules_column=num_rules_column,
        max_edit_rate=max_edit_rate,
        min_length_ratio=min_length_ratio,
        max_length_ratio=max_length_ratio,
        keep_unchanged=False,
        allow_new_adjacent_duplicates=True,
        drop_model_errors=True,
        drop_semantic_errors=True,
    )
    pairs_df = build_caa_pairs_dataframe(
        filtered_df,
        dialect=dialect,
        transform_model=transform_model,
        semantic_model=semantic_model,
        generation_config_json=generation_config_json,
        text_column=text_column,
        transformed_column=transformed_column,
        applied_rules_column=applied_rules_column,
        num_rules_column=num_rules_column,
    )

    audit_path = os.path.join(output_dir, f"{file_prefix}_caa_filter_audit.csv")
    pairs_path = os.path.join(output_dir, f"{file_prefix}_caa_pairs.csv")
    dropped_path = os.path.join(output_dir, f"{file_prefix}_caa_dropped.csv")

    audit_df.to_csv(audit_path, index=False)
    pairs_df.to_csv(pairs_path, index=False)
    dropped_df.to_csv(dropped_path, index=False)

    return {
        "audit_path": audit_path,
        "pairs_path": pairs_path,
        "dropped_path": dropped_path,
        "input_rows": len(df),
        "kept_rows": len(pairs_df),
        "dropped_rows": len(dropped_df),
    }


def filter_cefr_csv(input_csv, output_dir=None, file_prefix=None, **filter_kwargs):
    input_csv = os.path.abspath(input_csv)
    output_dir = os.path.abspath(output_dir or os.path.dirname(input_csv))
    file_prefix = file_prefix or os.path.splitext(os.path.basename(input_csv))[0]

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)
    audit_df, filtered_df, dropped_df = filter_cefr_dataframe(df, **filter_kwargs)

    audit_path = os.path.join(output_dir, f"{file_prefix}_filter_audit.csv")
    filtered_path = os.path.join(output_dir, f"{file_prefix}_filtered.csv")
    dropped_path = os.path.join(output_dir, f"{file_prefix}_dropped.csv")

    audit_df.to_csv(audit_path, index=False)
    filtered_df.to_csv(filtered_path, index=False)
    dropped_df.to_csv(dropped_path, index=False)

    return {
        "input_path": input_csv,
        "audit_path": audit_path,
        "filtered_path": filtered_path,
        "dropped_path": dropped_path,
        "input_rows": len(df),
        "kept_rows": len(filtered_df),
        "dropped_rows": len(dropped_df),
    }
