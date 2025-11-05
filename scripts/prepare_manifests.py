# scripts/prepare_manifests.py

import argparse
import logging
import os

import pandas as pd

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# --- Configuration ---
# Column name expected in all input CSVs
KEY_COL = "VolumeName"
# Suffix to remove from the keys for matching
SUFFIX_TO_REMOVE = ".nii.gz"
# --- End Configuration ---


def generate_path(volumename: str, prefix: str, suffix: str) -> str:
    """
    Generates a clean, relative path.
    e.g., ("train_123", "visual/", ".pt") -> "visual/train_123.pt"
    """
    # Ensure prefix ends with a slash if it's not empty
    if prefix and not prefix.endswith(("/")):
        prefix = f"{prefix}/"
        
    return f"{prefix}{volumename}{suffix}"


def create_manifest(args):
    """
    Loads master labels, a single split file, and (optionally) reports.
    Merges them, generates relative paths for visual and a single text
    feature, and *preserves* the original report text columns.
    """
    
    # --- 1. Load Master Labels (Required) ---
    log.info(f"Loading master labels from: {args.labels_csv}")
    try:
        labels_df = pd.read_csv(args.labels_csv)
    except FileNotFoundError:
        log.error(f"Labels file not found: {args.labels_csv}")
        raise
    if KEY_COL not in labels_df.columns:
        log.error(f"Labels CSV must contain '{KEY_COL}' column.")
        raise ValueError("Invalid labels.csv format")
    
    # Prepare labels merge key
    labels_df["merge_key"] = labels_df[KEY_COL].str.replace(
        SUFFIX_TO_REMOVE, "", regex=False
    )

    # --- 2. Load Split File (Required) ---
    log.info(f"Loading split file from: {args.split_csv}")
    try:
        split_df = pd.read_csv(args.split_csv)
    except FileNotFoundError:
        log.error(f"Split file not found: {args.split_csv}")
        raise
    if KEY_COL not in split_df.columns:
        log.error(f"Split CSV must contain '{KEY_COL}' column.")
        raise ValueError("Invalid split.csv format")

    # Prepare split merge key
    split_df["merge_key"] = split_df[KEY_COL].str.replace(
        SUFFIX_TO_REMOVE, "", regex=False
    )
    # We only need the key from the split file for the merge
    split_keys_df = split_df[["merge_key"]].drop_duplicates()

    # --- 3. Load Reports File (Optional) ---
    reports_df = None
    report_text_cols = []
    if args.reports_csv:
        log.info(f"Loading reports from: {args.reports_csv}")
        try:
            reports_df = pd.read_csv(args.reports_csv)
            if KEY_COL not in reports_df.columns:
                log.error(f"Reports CSV must contain '{KEY_COL}' column.")
                raise ValueError("Invalid reports.csv format")
            
            # Prepare reports merge key
            reports_df["merge_key"] = reports_df[KEY_COL].str.replace(
                SUFFIX_TO_REMOVE, "", regex=False
            )
            
            # Identify all columns *except* the key
            report_text_cols = [col for col in reports_df.columns if col not in [KEY_COL]]
            
            reports_df = reports_df[report_text_cols].drop_duplicates(subset=["merge_key"])
            log.info(f"Identified report text columns to preserve: {report_text_cols}")
            
        except FileNotFoundError:
            log.warning(f"Reports file not found at {args.reports_csv}. "
                        "Proceeding without report text.")
        except Exception as e:
            log.warning(f"Failed to read reports CSV: {e}. "
                        "Proceeding without report text.")

    # --- 4. Merge DataFrames ---
    log.info("Merging labels and split file on the processed merge key")
    
    # Start with the inner join of labels and splits
    merged_df = pd.merge(
        labels_df,
        split_keys_df,
        on="merge_key",
        how="inner"
    )

    if reports_df is not None:
        log.info("Merging reports data...")
        # Left join to keep all samples, even if they don't have a report
        merged_df = pd.merge(
            merged_df,
            reports_df,
            on="merge_key",
            how="left"
        )

    if merged_df.empty:
        log.warning("Merge resulted in an empty DataFrame. Check keys between files.")
        return

    log.info(f"Merged manifest contains {len(merged_df)} samples.")

    # --- 5. Generate Feature Path Columns ---
    log.info("Generating feature path columns...")
    
    # We use the 'merge_key' (which has the suffix removed) as the volumename
    merged_df["volumename"] = merged_df["merge_key"]
    
    # Generate Visual Path
    merged_df[args.visual_path_col] = merged_df["volumename"].apply(
        lambda x: generate_path(x, args.visual_prefix, ".pt")
    )

    # Generate Text Path (if text prefix is provided)
    if args.text_prefix:
        log.info(f"Generating single text path with prefix: {args.text_prefix}")
        merged_df[args.text_path_col] = merged_df["volumename"].apply(
            lambda x: generate_path(x, args.text_prefix, ".pt")
        )

    # --- 6. Finalize Columns ---
    # Drop intermediate keys and original KEY_COL
    merged_df = merged_df.drop(columns=[KEY_COL, "merge_key"])
    
    # --- 7. Create Output Directory (if needed) ---
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Ensured output directory exists: {output_dir}")

    # --- 8. Save Final Manifest File ---
    try:
        # Re-order columns to be cleaner
        id_cols = ["volumename", args.visual_path_col]
        if args.text_prefix:
            id_cols.append(args.text_path_col)
        
        label_cols = [col for col in labels_df.columns if col not in [KEY_COL, "merge_key"]]
        
        # This saves: ids, paths, labels, AND original report text
        final_cols = id_cols + label_cols + report_text_cols
        # Ensure we only try to save columns that actually exist
        final_cols = [col for col in final_cols if col in merged_df.columns]
        
        merged_df[final_cols].to_csv(args.output_file, index=False)
        
        log.info(f"Successfully saved manifest ({len(merged_df)} samples) to: "
                 f"{args.output_file}")
    except Exception as e:
        log.error(f"Failed to save manifest to '{args.output_file}': {e}")
        
    log.info("Manifest creation complete for this split.")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Prepare a single, multimodal CT-RATE Benchmark manifest file."
    )
    # --- Input Files ---
    parser.add_argument(
        "--labels_csv",
        type=str,
        required=True,
        help="Path to the master labels CSV file (e.g., 'all_predicted_labels.csv').",
    )
    parser.add_argument(
        "--split_csv",
        type=str,
        required=True,
        help="Path to the CSV file defining a single split (e.g., 'train.csv').",
    )
    parser.add_argument(
        "--reports_csv",
        type=str,
        default=None,
        help="(Optional) Path to the reports CSV file (e.g., 'all_reports.csv').",
    )
    
    # --- Output File ---
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the final output manifest (e.g., 'manifests/train.csv').",
    )
    
    # --- Visual Path Generation ---
    parser.add_argument(
        "--visual_prefix",
        type=str,
        default="image",
        help="Relative path prefix for visual features (e.g., 'image').",
    )
    parser.add_argument(
        "--visual_path_col",
        type=str,
        default="visual_feature_path",
        help="Column name for the generated visual path.",
    )
    
    # --- Text Path Generation (Single) ---
    parser.add_argument(
        "--text_prefix",
        type=str,
        default=None,
        help="(Optional) Relative path prefix for text features (e.g., 'text').",
    )
    parser.add_argument(
        "--text_path_col",
        type=str,
        default="text_feature_path",
        help="Column name for the generated text path.",
    )
    
    args = parser.parse_args()
    
    if args.reports_csv is None:
        log.warning("--- No --reports_csv provided. "
                    "Manifest will be VISUAL-ONLY and will not contain report text. ---")
    elif args.text_prefix is None:
        log.warning("--- No --text_prefix provided. "
                    "Manifest will include report text but NO text *feature path*. ---")

    create_manifest(args)


if __name__ == "__main__":
    main()