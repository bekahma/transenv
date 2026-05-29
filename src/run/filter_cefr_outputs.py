import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils.cefr_filter import filter_cefr_csv, write_caa_pair_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Post-filter generated CEFR text transformation CSVs for CAA/vector extraction."
    )
    parser.add_argument("--input_csv", required=True, help="Generated CEFR transformation CSV to filter")
    parser.add_argument("--output_dir", default=None, help="Directory for filtered/audit CSVs; defaults to input CSV directory")
    parser.add_argument("--file_prefix", default=None, help="Output file prefix; defaults to input CSV stem")
    parser.add_argument("--text_column", default="orig_sentence", help="Original SAE text column")
    parser.add_argument("--transformed_column", default="transformed_text", help="Generated transformed text column")
    parser.add_argument("--applied_rules_column", default="applied_rules", help="JSON list column with applied rules")
    parser.add_argument("--changed_column", default="is_changed", help="Boolean changed flag column")
    parser.add_argument("--num_rules_column", default="num_applied_rules", help="Applied rule count column")
    parser.add_argument("--max_edit_rate", type=float, default=0.10, help="Drop rows whose word edit rate is above this threshold")
    parser.add_argument("--min_length_ratio", type=float, default=0.50, help="Drop rows shorter than this transformed/original character ratio")
    parser.add_argument("--max_length_ratio", type=float, default=1.50, help="Drop rows longer than this transformed/original character ratio")
    parser.add_argument("--keep_unchanged", action="store_true", help="Keep rows with no accepted transformation")
    parser.add_argument("--allow_new_adjacent_duplicates", action="store_true", help="Keep rows that introduce adjacent duplicate tokens")
    parser.add_argument("--caa_pairs", action="store_true", help="Write CAA-ready pair CSVs instead of the generic filtered CSV")
    parser.add_argument("--dialect", default="", help="Dialect label for --caa_pairs output")
    parser.add_argument("--transform_model", default="", help="Transformation model label for --caa_pairs output")
    parser.add_argument("--semantic_model", default="", help="Semantic checker model label for --caa_pairs output")
    parser.add_argument("--generation_config_json", default="{}", help="Generation config JSON string for --caa_pairs output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.caa_pairs:
        import pandas as pd

        result = write_caa_pair_files(
            pd.read_csv(args.input_csv),
            output_dir=args.output_dir or os.path.dirname(os.path.abspath(args.input_csv)),
            file_prefix=args.file_prefix or os.path.splitext(os.path.basename(args.input_csv))[0],
            dialect=args.dialect,
            transform_model=args.transform_model,
            semantic_model=args.semantic_model,
            generation_config_json=args.generation_config_json,
            max_edit_rate=args.max_edit_rate,
            min_length_ratio=args.min_length_ratio,
            max_length_ratio=args.max_length_ratio,
        )
        print(f"Input rows: {result['input_rows']}")
        print(f"CAA pairs: {result['kept_rows']}")
        print(f"Dropped rows: {result['dropped_rows']}")
        print(f"Audit CSV: {result['audit_path']}")
        print(f"CAA pairs CSV: {result['pairs_path']}")
        print(f"Dropped CSV: {result['dropped_path']}")
        return

    result = filter_cefr_csv(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        file_prefix=args.file_prefix,
        text_column=args.text_column,
        transformed_column=args.transformed_column,
        applied_rules_column=args.applied_rules_column,
        changed_column=args.changed_column,
        num_rules_column=args.num_rules_column,
        max_edit_rate=args.max_edit_rate,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        keep_unchanged=args.keep_unchanged,
        allow_new_adjacent_duplicates=args.allow_new_adjacent_duplicates,
    )

    print(f"Input rows: {result['input_rows']}")
    print(f"Kept rows: {result['kept_rows']}")
    print(f"Dropped rows: {result['dropped_rows']}")
    print(f"Audit CSV: {result['audit_path']}")
    print(f"Filtered CSV: {result['filtered_path']}")
    print(f"Dropped CSV: {result['dropped_path']}")


if __name__ == "__main__":
    main()
