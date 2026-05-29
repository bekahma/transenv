import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from framework.data_return import return_cefr_texts
from framework import transformation as transformation_module
from framework.transformation import introduces_blank
from run import main as run_main
from utils.cefr_texts import (
    DIAGNOSTIC_COUNT_KEYS,
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

    @unittest.skipUnless(has_hf_datasets(), "Hugging Face datasets is not installed")
    def test_resume_at_max_samples_returns_empty_dataset(self):
        config = dataset_config(
            os.path.join(FIXTURE_DIR, "cefr_texts_levels.csv"),
            text_column="text",
            input_cefr_levels="A1,A2",
        )
        dataset = load_cefr_text_dataset(config, generation_config(max_samples=2), start_idx=2)

        self.assertEqual(len(dataset), 0)

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

    def test_extracts_transformed_sentence_marker_variants(self):
        self.assertEqual(
            extract_transformed_sentence("Final Transformed Sentence: She a doctor."),
            "She a doctor.",
        )
        self.assertEqual(
            extract_transformed_sentence('**Final broken sentence:** "I have book."'),
            '"I have book."',
        )

    def test_sentence_chunking_preserves_separators(self):
        chunks = split_text_chunks("Hello there.  How are you?\nFine.", mode="sentence", max_chunk_words=80)

        self.assertEqual(
            chunks,
            [
                {"text": "Hello there.", "separator": "  "},
                {"text": "How are you?", "separator": "\n"},
                {"text": "Fine.", "separator": ""},
            ],
        )

    def test_sentence_chunking_splits_long_sentences(self):
        chunks = split_text_chunks("one two three four five six.", mode="sentence", max_chunk_words=3)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["text"], "one two three")
        self.assertEqual(chunks[1]["text"], "four five six.")

    def test_hybrid_chunking_keeps_short_rows_whole(self):
        chunks = split_text_chunks("Short row. Has two sentences.", mode="hybrid", sentence_chunk_min_words=100)

        self.assertEqual(chunks, [{"text": "Short row. Has two sentences.", "separator": ""}])

    def test_hybrid_chunking_splits_long_rows(self):
        text = " ".join(["word"] * 101) + ". Next sentence."
        chunks = split_text_chunks(text, mode="hybrid", sentence_chunk_min_words=100)

        self.assertGreater(len(chunks), 1)
        self.assertEqual(chunks[-1]["text"], "Next sentence.")

    def test_row_chunking_keeps_long_rows_whole(self):
        text = " ".join(["word"] * 250) + ". Next sentence."
        chunks = split_text_chunks(text, mode="row", sentence_chunk_min_words=100, max_chunk_words=20)

        self.assertEqual(chunks, [{"text": text, "separator": ""}])

    def test_detects_newly_introduced_blank(self):
        self.assertTrue(introduces_blank("This is fine.", "This is <blank>."))
        self.assertFalse(introduces_blank("This is <blank>.", "This is <blank>."))

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
        self.assertEqual(
            aggregated["chunks"],
            [
                {
                    "chunk_index": 0,
                    "orig_text": "Hello there.",
                    "transformed_text": "Hello there.",
                    "applied_rules": [],
                    "is_changed": False,
                    "separator": " ",
                },
                {
                    "chunk_index": 1,
                    "orig_text": "How are you?",
                    "transformed_text": "How you?",
                    "applied_rules": ["MISSING VERB"],
                    "is_changed": True,
                    "separator": "",
                },
            ],
        )

    def test_aggregate_without_rules_preserves_original_whitespace(self):
        chunks = [
            {"text": "Hello there.", "separator": " "},
            {"text": "How are you?", "separator": ""},
        ]
        results = [
            {
                "final_sentence": "Hello there.",
                "applied_rules": [],
                "transformed_sentences": [],
            },
            {
                "final_sentence": "How are you?",
                "applied_rules": [],
                "transformed_sentences": [],
            },
        ]

        original = " \nHello there. How are you?\n "
        aggregated = aggregate_chunk_results(original, chunks, results)

        self.assertEqual(aggregated["final_sentence"], original)
        self.assertEqual(aggregated["applied_rules"], [])

    def test_aggregates_repeated_rule_applications(self):
        chunks = [
            {"text": "One.", "separator": " "},
            {"text": "Two.", "separator": ""},
        ]
        results = [
            {
                "final_sentence": "One changed.",
                "whole_response": [],
                "mid_transformed_sentences": [],
                "judge_repsonse": [],
                "applied_rules": ["OMISSION OF PREPOSITION"],
                "transformed_sentences": ["One changed."],
            },
            {
                "final_sentence": "Two changed.",
                "whole_response": [],
                "mid_transformed_sentences": [],
                "judge_repsonse": [],
                "applied_rules": ["OMISSION OF PREPOSITION"],
                "transformed_sentences": ["Two changed."],
            },
        ]

        aggregated = aggregate_chunk_results("One. Two.", chunks, results)

        self.assertEqual(
            aggregated["applied_rules"],
            ["OMISSION OF PREPOSITION", "OMISSION OF PREPOSITION"],
        )

    def test_row_rule_budget_trims_across_chunks(self):
        chunks = [
            {"text": "One.", "separator": " "},
            {"text": "Two.", "separator": " "},
            {"text": "Three.", "separator": ""},
        ]
        results = [
            {
                "final_sentence": "One changed.",
                "applied_rules": ["R1"],
                "transformed_sentences": ["One changed."],
            },
            {
                "final_sentence": "Two changed twice.",
                "applied_rules": ["R2", "R3"],
                "transformed_sentences": ["Two changed once.", "Two changed twice."],
            },
            {
                "final_sentence": "Three changed.",
                "applied_rules": ["R4"],
                "transformed_sentences": ["Three changed."],
            },
        ]

        trimmed = run_main._trim_results_to_row_rule_budget(results, chunks, max_rules_per_row=2)
        aggregated = aggregate_chunk_results("One. Two. Three.", chunks, trimmed)

        self.assertEqual(trimmed[0]["applied_rules"], ["R1"])
        self.assertEqual(trimmed[1]["applied_rules"], ["R2"])
        self.assertEqual(trimmed[1]["final_sentence"], "Two changed once.")
        self.assertEqual(trimmed[2]["applied_rules"], [])
        self.assertEqual(aggregated["final_sentence"], "One changed. Two changed once. Three.")
        self.assertEqual(aggregated["applied_rules"], ["R1", "R2"])

    def test_chunk_and_row_rule_budgets_stop_later_chunks(self):
        chunks = [
            {"text": "One.", "separator": " "},
            {"text": "Two.", "separator": " "},
            {"text": "Three.", "separator": ""},
        ]
        seen_chunk_budgets = []

        def fake_transform(sentences, *args, **kwargs):
            self.assertEqual(len(sentences), 1)
            seen_chunk_budgets.append(kwargs["max_rules_per_chunk"])
            text = sentences[0]
            return [
                {
                    "orig_sentence": text,
                    "final_sentence": f"{text} changed",
                    "whole_response": [],
                    "mid_transformed_sentences": [],
                    "judge_repsonse": [],
                    "applied_rules": [f"rule:{text}"],
                    "transformed_sentences": [f"{text} changed"],
                }
            ]

        with patch.object(run_main, "_transform_sentences", side_effect=fake_transform):
            results = run_main._transform_cefr_chunks(
                chunks,
                guideline=[],
                client=None,
                tokenizer=None,
                sampling_params={},
                task_config=None,
                model_config=None,
                use_hosted_openai=True,
                batch_size=5,
                max_rules_per_chunk=1,
                max_rules_per_row=2,
            )

        self.assertEqual(seen_chunk_budgets, [1, 1])
        self.assertEqual(results[0]["applied_rules"], ["rule:One."])
        self.assertEqual(results[1]["applied_rules"], ["rule:Two."])
        self.assertEqual(results[2]["final_sentence"], "Three.")
        self.assertEqual(results[2]["applied_rules"], [])

    def test_hosted_row_budget_batches_chunks_when_parallelism_enabled(self):
        chunks = [
            {"text": "One.", "separator": " "},
            {"text": "Two.", "separator": " "},
            {"text": "Three.", "separator": ""},
        ]
        calls = []

        def fake_transform(sentences, *args, **kwargs):
            calls.append((list(sentences), kwargs["max_rules_per_chunk"], kwargs["openai_parallelism"]))
            return [
                {
                    "orig_sentence": text,
                    "final_sentence": f"{text} changed",
                    "whole_response": [],
                    "mid_transformed_sentences": [],
                    "judge_repsonse": [],
                    "applied_rules": [f"rule:{text}"],
                    "transformed_sentences": [f"{text} changed"],
                }
                for text in sentences
            ]

        with patch.object(run_main, "_transform_sentences", side_effect=fake_transform):
            results = run_main._transform_cefr_chunks(
                chunks,
                guideline=[],
                client=None,
                tokenizer=None,
                sampling_params={},
                task_config=None,
                model_config=None,
                use_hosted_openai=True,
                batch_size=5,
                max_rules_per_chunk=1,
                max_rules_per_row=2,
                openai_parallelism=2,
            )

        self.assertEqual(calls, [(["One.", "Two."], 1, 2)])
        self.assertEqual(results[0]["applied_rules"], ["rule:One."])
        self.assertEqual(results[1]["applied_rules"], ["rule:Two."])
        self.assertEqual(results[2]["final_sentence"], "Three.")

    def test_row_chunk_batch_transforms_rows_together(self):
        calls = []

        def fake_transform(sentences, *args, **kwargs):
            calls.append((list(sentences), kwargs["max_rules_per_chunk"], kwargs["openai_parallelism"]))
            return [
                {
                    "orig_sentence": text,
                    "final_sentence": f"{text} changed",
                    "whole_response": [],
                    "mid_transformed_sentences": [],
                    "judge_repsonse": [],
                    "applied_rules": [f"rule:{text}"],
                    "transformed_sentences": [f"{text} changed"],
                }
                for text in sentences
            ]

        config = SimpleNamespace(
            batch_size=5,
            max_rules_per_chunk=2,
            max_rules_per_row=1,
            openai_parallelism=3,
        )

        with patch.object(run_main, "_transform_sentences", side_effect=fake_transform):
            outputs = run_main._transform_cefr_row_batch(
                ["One.", "Two.", ""],
                [False, False, True],
                [10, 11, 12],
                guideline=[],
                client=None,
                tokenizer=None,
                sampling_params={},
                task_config=None,
                model_config=None,
                use_hosted_openai=True,
                generation_config=config,
            )

        self.assertEqual(calls, [(["One.", "Two."], 1, 3)])
        self.assertEqual(outputs[0]["final_sentence"], "One. changed")
        self.assertEqual(outputs[1]["applied_rules"], ["rule:Two."])
        self.assertEqual(outputs[2]["final_sentence"], "")

    def test_three_rule_budget_trims_extra_applications(self):
        result = {
            "final_sentence": "fourth",
            "applied_rules": ["R1", "R2", "R3", "R4"],
            "transformed_sentences": ["first", "second", "third", "fourth"],
        }

        trimmed = run_main._limit_result_to_rule_budget(result, "original", max_rules=3)

        self.assertEqual(trimmed["applied_rules"], ["R1", "R2", "R3"])
        self.assertEqual(trimmed["final_sentence"], "third")
        self.assertTrue(trimmed["rule_limit_truncated"])

    def test_generation_limits_reject_zero_rule_budgets(self):
        config = SimpleNamespace(
            batch_size=1,
            openai_parallelism=1,
            max_samples=None,
            max_rules_per_chunk=0,
            max_rules_per_row=None,
        )

        with self.assertRaisesRegex(ValueError, "max_rules_per_chunk"):
            run_main._validate_generation_limits(config)

        config.max_rules_per_chunk = None
        config.max_rules_per_row = 0
        with self.assertRaisesRegex(ValueError, "max_rules_per_row"):
            run_main._validate_generation_limits(config)

        config.max_rules_per_row = None
        config.openai_parallelism = 0
        with self.assertRaisesRegex(ValueError, "openai_parallelism"):
            run_main._validate_generation_limits(config)

    def test_all_model_errors_raise_clear_runtime_error(self):
        output = {
            "whole_response": ["rate limit", "rate limit", "rate limit", "rate limit", "rate limit"],
            "model_errors": ["rate limit"],
            "model_error_count": 5,
        }

        with self.assertRaisesRegex(RuntimeError, "All hosted OpenAI transformation calls failed"):
            run_main._raise_if_all_model_calls_failed(0, output)

    def test_hosted_transformation_aborts_repeated_api_errors(self):
        class FailingCompletions:
            def create(self, *args, **kwargs):
                raise RuntimeError("rate limit")

        client = SimpleNamespace(chat=SimpleNamespace(completions=FailingCompletions()))
        task_config = SimpleNamespace(task_name="english_dialect")
        model_config = SimpleNamespace(model_name="dummy", semantic_model_name=None)
        sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 20}
        guideline = [(f"feature {idx}", "unused") for idx in range(10)]

        with patch.object(
            transformation_module,
            "openai_framework_application",
            return_value=[{"role": "system", "content": "system"}],
        ):
            with self.assertRaisesRegex(RuntimeError, "failed 10 times in a row"):
                transformation_module.openai_transformation(
                    ["Sentence."],
                    guideline,
                    client,
                    sampling_params,
                    task_config,
                    model_config,
                )

    def test_hosted_transformation_can_use_openai_batch_api(self):
        class FakeFiles:
            def __init__(self):
                self.contents = {}
                self.next_file = 0

            def create(self, file, purpose):
                self.next_file += 1
                return SimpleNamespace(id=f"file-input-{self.next_file}")

            def content(self, file_id):
                return SimpleNamespace(text=self.contents[file_id])

        class FakeBatches:
            def __init__(self, files):
                self.files = files
                self.next_batch = 0

            def create(self, input_file_id, endpoint, completion_window):
                self.next_batch += 1
                batch_id = f"batch-{self.next_batch}"
                output_file_id = f"file-output-{self.next_batch}"
                if self.next_batch == 1:
                    response = "**Transformed Sentence:** She a teacher."
                    custom_id = "transform:0:0:row:0"
                else:
                    response = "no"
                    custom_id = "semantic:0:0:row:0"
                self.files.contents[output_file_id] = json.dumps({
                    "custom_id": custom_id,
                    "response": {
                        "status_code": 200,
                        "body": {"choices": [{"message": {"content": response}}]},
                    },
                    "error": None,
                }) + "\n"
                return SimpleNamespace(id=batch_id, status="completed", output_file_id=output_file_id, error_file_id=None)

            def retrieve(self, batch_id):
                raise AssertionError("completed fake batches should not be polled")

        files = FakeFiles()
        client = SimpleNamespace(files=files, batches=FakeBatches(files))
        task_config = SimpleNamespace(task_name="english_dialect")
        model_config = SimpleNamespace(model_name="gpt-4.1-mini", semantic_model_name=None)
        sampling_params = {"temperature": 0.8, "top_p": 0.95, "max_tokens": 20}

        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch.object(
                transformation_module,
                "openai_framework_application",
                return_value=[{"role": "system", "content": "system"}],
            ):
                output = transformation_module.openai_transformation(
                    ["She is a teacher."],
                    [("Deletion of copula be: before NPs", "unused")],
                    client,
                    sampling_params,
                    task_config,
                    model_config,
                    max_rules_per_chunk=1,
                    openai_call_mode="batch",
                    openai_batch_output_dir=tmp_dir,
                    openai_batch_poll_interval=1,
                )[0]

        self.assertEqual(output["final_sentence"], "She a teacher.")
        self.assertEqual(output["applied_rules"], ["Deletion of copula be: before NPs"])
        self.assertEqual(output["judge_repsonse"], ["no"])

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
                        "whole_response": ["response"],
                        "mid_transformed_sentences": ["I likes apples."],
                        "judge_repsonse": ["no"],
                        "model_errors": [],
                        "semantic_errors": [],
                        "model_error_count": 0,
                        "semantic_error_count": 0,
                        "no_change_response_count": 3,
                        "parse_failure_count": 1,
                        "blank_rejection_count": 0,
                        "semantic_rejection_count": 0,
                        "applied_rules": ["SUBJECT VERB AGREEMENT"],
                        "chunk_count": 2,
                        "chunks": [
                            {
                                "chunk_index": 0,
                                "orig_text": "I like apples.",
                                "transformed_text": "I likes apples.",
                                "applied_rules": ["SUBJECT VERB AGREEMENT"],
                                "is_changed": True,
                                "separator": "",
                            },
                            {
                                "chunk_index": 1,
                                "orig_text": "No change.",
                                "transformed_text": "No change.",
                                "applied_rules": [],
                                "is_changed": False,
                                "separator": "",
                            },
                        ],
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
        self.assertIn("changed_chunk_count", saved.columns)
        self.assertIn("changed_chunk_indices", saved.columns)
        self.assertIn("changed_chunks", saved.columns)
        self.assertIn("model_response_count", saved.columns)
        self.assertIn("candidate_transform_count", saved.columns)
        self.assertIn("semantic_judge_count", saved.columns)
        for key in DIAGNOSTIC_COUNT_KEYS:
            self.assertIn(key, saved.columns)
        self.assertIn("sample_model_error", saved.columns)
        self.assertIn("sample_semantic_error", saved.columns)
        self.assertIn("sample_model_response", saved.columns)
        self.assertIn("source_row_idx", saved.columns)
        self.assertEqual(saved.loc[0, "transformed_text"], "I likes apples.")
        self.assertEqual(saved.loc[0, "source_row_idx"], 0)
        self.assertEqual(json.loads(saved.loc[0, "applied_rules"]), ["SUBJECT VERB AGREEMENT"])
        self.assertEqual(saved.loc[0, "num_applied_rules"], 1)
        self.assertTrue(bool(saved.loc[0, "is_changed"]))
        self.assertEqual(saved.loc[0, "changed_chunk_count"], 1)
        self.assertEqual(saved.loc[0, "model_response_count"], 1)
        self.assertEqual(saved.loc[0, "candidate_transform_count"], 1)
        self.assertEqual(saved.loc[0, "semantic_judge_count"], 1)
        self.assertEqual(saved.loc[0, "no_change_response_count"], 3)
        self.assertEqual(saved.loc[0, "parse_failure_count"], 1)
        self.assertEqual(saved.loc[0, "sample_model_response"], "response")
        self.assertEqual(json.loads(saved.loc[0, "changed_chunk_indices"]), [0])
        self.assertEqual(
            json.loads(saved.loc[0, "changed_chunks"]),
            [
                {
                    "chunk_index": 0,
                    "orig_text": "I like apples.",
                    "transformed_text": "I likes apples.",
                    "applied_rules": ["SUBJECT VERB AGREEMENT"],
                }
            ],
        )
        self.assertFalse(bool(saved.loc[1, "is_changed"]))
        self.assertEqual(saved.loc[1, "changed_chunk_count"], 0)

    def test_writes_caa_pair_export_when_enabled(self):
        config = dataset_config(
            os.path.join(FIXTURE_DIR, "cefr_texts_levels.csv"),
            text_column="text",
            input_cefr_levels="A1,A2",
        )
        gen_config = SimpleNamespace(
            batch_size=2,
            rerun=None,
            max_samples=None,
            write_caa_pairs=True,
            caa_max_edit_rate=0.50,
            caa_min_length_ratio=0.50,
            caa_max_length_ratio=1.80,
        )
        task_config = SimpleNamespace(dialect="Urban African American Vernacular English")
        model_config = SimpleNamespace(model_name="gpt-4.1-mini", semantic_model_name=None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_config = SimpleNamespace(save_path=tmp_dir, file_name="aave_row")
            outputs = {
                "question": [
                    {
                        "orig_sentence": "I like those apples every day.",
                        "final_sentence": "I likes them apples every day.",
                        "whole_response": ["response"],
                        "mid_transformed_sentences": ["I likes them apples every day."],
                        "judge_repsonse": ["no"],
                        "model_errors": [],
                        "semantic_errors": [],
                        "model_error_count": 0,
                        "semantic_error_count": 0,
                        "no_change_response_count": 0,
                        "parse_failure_count": 0,
                        "blank_rejection_count": 0,
                        "semantic_rejection_count": 0,
                        "applied_rules": ["SUBJECT VERB AGREEMENT", "THEM INSTEAD OF THOSE"],
                        "transformed_sentences": ["I likes those apples every day.", "I likes them apples every day."],
                    },
                    {
                        "orig_sentence": "The train arrived late.",
                        "final_sentence": "The train arrived late.",
                        "applied_rules": [],
                    },
                ]
            }

            return_cefr_texts(outputs, save_config, config, gen_config, task_config, model_config)
            pairs = pd.read_csv(os.path.join(tmp_dir, "aave_row_caa_pairs.csv"))
            audit = pd.read_csv(os.path.join(tmp_dir, "aave_row_caa_filter_audit.csv"))

        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs.loc[0, "source_row_idx"], 0)
        self.assertEqual(pairs.loc[0, "dialect"], "Urban African American Vernacular English")
        self.assertEqual(pairs.loc[0, "transform_model"], "gpt-4.1-mini")
        self.assertEqual(json.loads(pairs.loc[0, "applied_rules"]), ["SUBJECT VERB AGREEMENT", "THEM INSTEAD OF THOSE"])
        self.assertEqual(pairs.loc[0, "num_applied_rules"], 2)
        self.assertIn("generation_config_json", pairs.columns)
        self.assertEqual(len(audit), 2)


if __name__ == "__main__":
    unittest.main()
