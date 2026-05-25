import json
import os
import sys
import tempfile
import unittest

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.cefr_filter import filter_cefr_csv, filter_cefr_dataframe


def make_text(prefix, count=30):
    return " ".join(f"{prefix}{idx}" for idx in range(count))


class CefrFilterTest(unittest.TestCase):
    def test_filter_flags_outliers_and_keeps_clean_pairs(self):
        clean_orig = make_text("word")
        clean_transformed = clean_orig.replace("word0 ", "", 1)

        df = pd.DataFrame(
            [
                {
                    "orig_sentence": clean_orig,
                    "transformed_text": clean_transformed,
                    "applied_rules": json.dumps(["MISSING SUBJECT"]),
                    "num_applied_rules": 1,
                    "is_changed": True,
                },
                {
                    "orig_sentence": make_text("same"),
                    "transformed_text": make_text("same"),
                    "applied_rules": json.dumps([]),
                    "num_applied_rules": 0,
                    "is_changed": False,
                },
                {
                    "orig_sentence": "This sentence has a good idea for class.",
                    "transformed_text": "This sentence has a good idea idea for class.",
                    "applied_rules": json.dumps(["SHOULD BE PLURAL"]),
                    "num_applied_rules": 1,
                    "is_changed": True,
                },
                {
                    "orig_sentence": make_text("long"),
                    "transformed_text": "short text",
                    "applied_rules": json.dumps(["MISSING SUBJECT"]),
                    "num_applied_rules": 1,
                    "is_changed": True,
                },
                {
                    "orig_sentence": "This source has no blank token.",
                    "transformed_text": "This source has <blank> token.",
                    "applied_rules": json.dumps(["MISSING VERB"]),
                    "num_applied_rules": 1,
                    "is_changed": True,
                },
            ]
        )

        audit_df, filtered_df, dropped_df = filter_cefr_dataframe(df, max_edit_rate=0.10)

        self.assertEqual(len(audit_df), 5)
        self.assertEqual(len(filtered_df), 1)
        self.assertEqual(len(dropped_df), 4)
        self.assertTrue(bool(audit_df.loc[0, "filter_keep"]))
        self.assertEqual(audit_df.loc[0, "transformed_text_filtered"], clean_transformed)
        self.assertFalse(bool(audit_df.loc[1, "filter_keep"]))
        self.assertIn("unchanged_or_no_applied_rules", audit_df.loc[1, "filter_reasons"])
        self.assertIn("new_adjacent_duplicate_tokens", audit_df.loc[2, "filter_reasons"])
        self.assertIn("edit_rate_gt_0.10", audit_df.loc[3, "filter_reasons"])
        self.assertIn("introduced_blank", audit_df.loc[4, "filter_reasons"])

    def test_filter_csv_writes_audit_filtered_and_dropped_outputs(self):
        df = pd.DataFrame(
            [
                {
                    "orig_sentence": make_text("word"),
                    "transformed_text": make_text("word").replace("word0 ", "", 1),
                    "applied_rules": json.dumps(["MISSING SUBJECT"]),
                    "num_applied_rules": 1,
                    "is_changed": True,
                },
                {
                    "orig_sentence": make_text("same"),
                    "transformed_text": make_text("same"),
                    "applied_rules": json.dumps([]),
                    "num_applied_rules": 0,
                    "is_changed": False,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "generated.csv")
            df.to_csv(input_path, index=False)

            result = filter_cefr_csv(input_path, output_dir=tmp_dir, file_prefix="generated")

            self.assertTrue(os.path.exists(result["audit_path"]))
            self.assertTrue(os.path.exists(result["filtered_path"]))
            self.assertTrue(os.path.exists(result["dropped_path"]))

            audit = pd.read_csv(result["audit_path"])
            filtered = pd.read_csv(result["filtered_path"])
            dropped = pd.read_csv(result["dropped_path"])

        self.assertEqual(len(audit), 2)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(len(dropped), 1)
        self.assertIn("transformed_text", audit.columns)
        self.assertIn("transformed_text_filtered", audit.columns)


if __name__ == "__main__":
    unittest.main()
