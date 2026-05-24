import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from framework.data_return import return_cefr_texts
from utils.cefr_texts import (
    INTERNAL_EMPTY_TEXT_COLUMN,
    INTERNAL_ROW_INDEX_COLUMN,
    INTERNAL_TEXT_COLUMN,
    aggregate_chunk_results,
    detect_text_column,
    empty_transformation_result,
    load_cefr_text_dataset,
    parse_cefr_levels,
    split_text_chunks,
)
from utils.guidline_utils import extract_transformed_sentence


FIXTURE_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def has_hf_datasets():
    try:
        import datasets

        return hasattr(datasets, "load_dataset")
    except Exception:
        return False


def dataset_config(input_path, text_column=None, input_cefr_levels=None):
    return SimpleNamespace(
        dataset_name="cefr_texts",
        input_path=input_path,
        text_column=text_column,
        input_cefr_levels=input_cefr_levels,
    )


def generation_config(batch_size=2, rerun=None, max_samples=None):
    return SimpleNamespace(batch_size=batch_size, rerun=rerun, max_samples=max_samples)


class CefrTextsTest(unittest.TestCase):
    @unittest.skipUnless(has_hf_datasets(), "Hugging Face datasets is not installed")
    def test_detects_single_text_column_and_resume_skip(self):
        config = dataset_config(os.path.join(FIXTURE_DIR, "cefr_texts_one_column.csv"))
        dataset = load_cefr_text_dataset(config, generation_config(), start_idx=1)

        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset[0][INTERNAL_TEXT_COLUMN], "The rain stopped before lunch.")
        self.assertFalse(dataset[0][INTERNAL_EMPTY_TEXT_COLUMN])
        self.assertEqual(dataset[0][INTERNAL_ROW_INDEX_COLUMN], 1)

    @unittest.skipUnless(has_hf_datasets(), "Hugging Face datasets is not installed")
    def test_filters_input_cefr_levels(self):
        config = dataset_config(
            os.path.join(FIXTURE_DIR, "cefr_texts_levels.csv"),
            text_column="text",
            input_cefr_levels="A1,A2",
        )
        dataset = load_cefr_text_dataset(config, generation_config())

        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset[0][INTERNAL_ROW_INDEX_COLUMN], 0)
        self.assertEqual(dataset[1][INTERNAL_ROW_INDEX_COLUMN], 1)
        self.assertEqual(dataset[2][INTERNAL_ROW_INDEX_COLUMN], 3)
        self.assertTrue(dataset[2][INTERNAL_EMPTY_TEXT_COLUMN])

    @unittest.skipUnless(has_hf_datasets(), "Hugging Face datasets is not installed")
    def test_limits_max_samples_after_filtering(self):
        config = dataset_config(
            os.path.join(FIXTURE_DIR, "cefr_texts_levels.csv"),
            text_column="text",
            input_cefr_levels="A1,A2",
        )
        dataset = load_cefr_text_dataset(config, generation_config(max_samples=2))

        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0][INTERNAL_ROW_INDEX_COLUMN], 0)
        self.assertEqual(dataset[1][INTERNAL_ROW_INDEX_COLUMN], 1)

    def test_cefr_group_shorthand(self):
        self.assertEqual(parse_cefr_levels("A"), {"A1", "A2"})
        self.assertEqual(parse_cefr_levels("B1,C"), {"B1", "C1", "C2"})

    def test_requires_text_column_when_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "Pass --text_column"):
            detect_text_column(["title", "body"])

    def test_extracts_multiline_transformed_sentence(self):
        response = """Phase 1: applicable

**Transformed Sentence:**
Line one.
Line two."""

        self.assertEqual(extract_transformed_sentence(response), "Line one.\nLine two.")

    def test_empty_transformed_sentence_is_no_change(self):
        self.assertEqual(extract_transformed_sentence("**Transformed Sentence:**\n"), "No change")

    def test_sentence_chunking_preserves_separators(self):
        chunks = split_text_chunks("Hello there.  How are you?\nFine.", max_chunk_words=80)

        self.assertEqual(
            chunks,
            [
                {"text": "Hello there.", "separator": "  "},
                {"text": "How are you?", "separator": "\n"},
                {"text": "Fine.", "separator": ""},
            ],
        )

    def test_sentence_chunking_splits_long_sentences(self):
        chunks = split_text_chunks("one two three four five six.", max_chunk_words=3)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "one two three")
        self.assertEqual(chunks[1]["text"], "four five six.")

    def test_aggregates_chunk_results(self):
        chunks = [
            {"text": "Hello there.", "separator": " "},
            {"text": "How are you?", "separator": ""},
        ]
        results = [
            {
                "final_sentence": "Hello there.",
                "whole_response": ["r1"],
                "mid_transformed_sentences": [],
                "judge_repsonse": [],
                "applied_rules": [],
                "transformed_sentences": [],
            },
            {
                "final_sentence": "How you?",
                "whole_response": ["r2"],
                "mid_transformed_sentences": ["How you?"],
                "judge_repsonse": ["no"],
                "applied_rules": ["MISSING VERB"],
                "transformed_sentences": ["How you?"],
            },
        ]

        aggregated = aggregate_chunk_results("Hello there. How are you?", chunks, results)

        self.assertEqual(aggregated["final_sentence"], "Hello there. How you?")
        self.assertEqual(aggregated["applied_rules"], ["MISSING VERB"])
        self.assertEqual(aggregated["chunk_count"], 2)

    def test_saves_appended_audit_columns(self):
        config = dataset_config(
            os.path.join(FIXTURE_DIR, "cefr_texts_levels.csv"),
            text_column="text",
            input_cefr_levels="A1,A2",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_config = SimpleNamespace(save_path=tmp_dir, file_name="A_arabic")
            outputs = {
                "question": [
                    {
                        "orig_sentence": "I like apples.",
                        "final_sentence": "I likes apples.",
                        "applied_rules": ["SUBJECT VERB AGREEMENT"],
                    },
                    {
                        "orig_sentence": "The train arrived late.",
                        "final_sentence": "The train arrived late.",
                        "applied_rules": [],
                    },
                    empty_transformation_result(""),
                ]
            }

            return_cefr_texts(outputs, save_config, config, generation_config())
            saved = pd.read_csv(os.path.join(tmp_dir, "A_arabic.csv"))

        self.assertEqual(len(saved), 3)
        self.assertIn("transformed_text", saved.columns)
        self.assertIn("applied_rules", saved.columns)
        self.assertIn("num_applied_rules", saved.columns)
        self.assertIn("is_changed", saved.columns)
        self.assertIn("chunk_count", saved.columns)
        self.assertEqual(saved.loc[0, "transformed_text"], "I likes apples.")
        self.assertEqual(json.loads(saved.loc[0, "applied_rules"]), ["SUBJECT VERB AGREEMENT"])
        self.assertEqual(saved.loc[0, "num_applied_rules"], 1)
        self.assertTrue(bool(saved.loc[0, "is_changed"]))
        self.assertFalse(bool(saved.loc[1, "is_changed"]))


if __name__ == "__main__":
    unittest.main()
