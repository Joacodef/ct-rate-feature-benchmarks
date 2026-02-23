# scripts/prepare_manifests.py

"""Prepare multimodal training/evaluation manifests from source CSV inputs.

The script merges label declarations with split membership and optional
report text, generates relative feature paths, and writes a final manifest
with stable column ordering.
"""

import argparse
import logging
import os

import pandas as pd

# Configure basic logging.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# Column name expected in all input CSVs.
KEY_COL = "VolumeName"
# Suffix removed from keys for merge normalization.
SUFFIX_TO_REMOVE = ".nii.gz"


def generate_path(volumename: str, prefix: str, suffix: str) -> str:
    """Generate a normalized relative feature path.

    Args:
        volumename: Volume identifier used as filename stem.
        prefix: Relative path prefix (e.g., ``image`` or ``text``).
        suffix: File suffix/extension.

    Returns:
        Relative feature path (e.g., ``image/train_123.pt``).

    Logic:
        Ensure ``prefix`` ends with ``/`` when non-empty and ensure
        ``suffix`` starts with ``.`` when non-empty.
    """
    # Normalize prefix.
    if prefix and not prefix.endswith(("/")):
        prefix = f"{prefix}/"

    # Normalize suffix.
    if suffix and not suffix.startswith("."):
        suffix = f".{suffix}"

    return f"{prefix}{volumename}{suffix}"


def normalize_key(value: str) -> str:
    """Normalize a key by removing the known volume suffix when present.

    Args:
        value: Raw key value.

    Returns:
        Normalized key string or original value when not applicable.
    """
    if not isinstance(value, str):
        return value
    if value.endswith(SUFFIX_TO_REMOVE):
        return value[: -len(SUFFIX_TO_REMOVE)]
    return value


def base_key_from_full(full_key: str) -> str:
    """Derive base exam key by stripping the final reconstruction segment.

    Args:
        full_key: Reconstruction-level key (e.g., ``train_1_a_2``).

    Returns:
        Base key (e.g., ``train_1_a``) or original value when unavailable.
    """
    if not isinstance(full_key, str):
        return full_key
    if "_" not in full_key:
        return full_key
    return full_key.rsplit("_", 1)[0]


def create_manifest(args: argparse.Namespace) -> None:
    """Create a merged manifest CSV for one split.

    Args:
        args: Parsed CLI arguments containing input paths and output schema.

    Returns:
        ``None``.

    Logic:
        Load and normalize label/split sources, optionally expand labels using
        metadata and merge report text, generate feature-path columns, then
        write a clean output CSV.
    """

    # 1) Load master labels.
    log.info(f"Loading master labels from: {args.labels_csv}")
    try:
        labels_df = pd.read_csv(args.labels_csv)
    except FileNotFoundError:
        log.error(f"Labels file not found: {args.labels_csv}")
        raise
    if KEY_COL not in labels_df.columns:
        log.error(f"Labels CSV must contain '{KEY_COL}' column.")
        raise ValueError("Invalid labels.csv format")
    
    # Build label merge key.
    labels_df["merge_key"] = labels_df[KEY_COL].apply(normalize_key)

    # 1b) Optionally expand labels to reconstruction-level keys.
    if args.metadata_csv:
        log.info(f"Loading metadata for reconstruction expansion: {args.metadata_csv}")
        try:
            metadata_df = pd.read_csv(args.metadata_csv, usecols=[KEY_COL])
        except FileNotFoundError:
            log.error(f"Metadata file not found: {args.metadata_csv}")
            raise
        if KEY_COL not in metadata_df.columns:
            log.error(f"Metadata CSV must contain '{KEY_COL}' column.")
            raise ValueError("Invalid metadata.csv format")

        metadata_keys_df = metadata_df[[KEY_COL]].copy()
        metadata_keys_df["full_key"] = metadata_keys_df[KEY_COL].apply(normalize_key)
        metadata_keys_df["base_key"] = metadata_keys_df["full_key"].apply(base_key_from_full)
        metadata_keys_df = metadata_keys_df[["base_key", "full_key"]].drop_duplicates()

        expanded_df = pd.merge(
            labels_df,
            metadata_keys_df,
            left_on="merge_key",
            right_on="base_key",
            how="left",
        )

        missing_expansions = expanded_df["full_key"].isna().sum()
        if missing_expansions:
            log.warning(
                "%s labels could not be expanded via metadata; using original keys.",
                missing_expansions,
            )

        expanded_df["merge_key"] = expanded_df["full_key"].fillna(expanded_df["merge_key"])
        labels_df = expanded_df.drop(columns=["base_key", "full_key"])

    # 2) Load split file.
    log.info(f"Loading split file from: {args.split_csv}")
    try:
        split_df = pd.read_csv(args.split_csv)
    except FileNotFoundError:
        log.error(f"Split file not found: {args.split_csv}")
        raise
    if KEY_COL not in split_df.columns:
        log.error(f"Split CSV must contain '{KEY_COL}' column.")
        raise ValueError("Invalid split.csv format")

    # Build split merge key.
    split_df["merge_key"] = split_df[KEY_COL].apply(normalize_key)
    # Keep only unique split keys for membership filtering.
    split_keys_df = split_df[["merge_key"]].drop_duplicates()

    # 3) Optionally load reports.
    reports_df = None
    report_text_cols = []
    if args.reports_csv:
        log.info(f"Loading reports from: {args.reports_csv}")
        try:
            reports_df = pd.read_csv(args.reports_csv)
            if KEY_COL not in reports_df.columns:
                log.error(f"Reports CSV must contain '{KEY_COL}' column.")
                raise ValueError("Invalid reports.csv format")
            
            # Build reports merge key.
            reports_df["merge_key"] = reports_df[KEY_COL].apply(normalize_key)
            
            # Preserve non-key report text columns.
            report_text_cols = [col for col in reports_df.columns if col not in [KEY_COL]]
            
            reports_df = reports_df[report_text_cols].drop_duplicates(subset=["merge_key"])
            log.info(f"Identified report text columns to preserve: {report_text_cols}")
            
        except FileNotFoundError:
            log.warning(f"Reports file not found at {args.reports_csv}. "
                        "Proceeding without report text.")
        except Exception as e:
            log.warning(f"Failed to read reports CSV: {e}. "
                        "Proceeding without report text.")

    # 4) Merge datasets.
    log.info("Merging labels and split file on the processed merge key")

    # Start with labels restricted to split membership.
    merged_df = pd.merge(
        labels_df,
        split_keys_df,
        on="merge_key",
        how="inner"
    )

    if reports_df is not None:
        log.info("Merging reports data...")
        # Keep all samples even if report text is missing.
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

    # 5) Generate feature-path columns.
    log.info("Generating feature path columns...")

    # Use normalized merge key as canonical volumename.
    merged_df["volumename"] = merged_df["merge_key"]

    # Generate visual feature path.
    merged_df[args.visual_path_col] = merged_df["volumename"].apply(
        lambda x: generate_path(x, args.visual_prefix, args.visual_suffix if hasattr(args, 'visual_suffix') else ".pt")
    )

    # Generate text feature path when configured.
    if args.text_prefix:
        log.info(f"Generating single text path with prefix: {args.text_prefix}")
        merged_df[args.text_path_col] = merged_df["volumename"].apply(
            lambda x: generate_path(x, args.text_prefix, args.text_suffix if hasattr(args, 'text_suffix') else ".pt")
        )

    # 6) Finalize dataframe columns.
    # Drop intermediate keys and original key column.
    merged_df = merged_df.drop(columns=[KEY_COL, "merge_key"])

    # 7) Ensure output directory exists.
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        log.info(f"Ensured output directory exists: {output_dir}")

    # 8) Write final manifest file.
    try:
        # Order columns as IDs/paths, labels, then report text.
        id_cols = ["volumename", args.visual_path_col]
        if args.text_prefix:
            id_cols.append(args.text_path_col)
        
        label_cols = [col for col in labels_df.columns if col not in [KEY_COL, "merge_key"]]
        
        final_cols = id_cols + label_cols + report_text_cols
        # Keep only columns that exist after merges.
        final_cols = [col for col in final_cols if col in merged_df.columns]
        
        merged_df[final_cols].to_csv(args.output_file, index=False)
        
        log.info(f"Successfully saved manifest ({len(merged_df)} samples) to: "
                 f"{args.output_file}")
    except Exception as e:
        log.error(f"Failed to save manifest to '{args.output_file}': {e}")
        
    log.info("Manifest creation complete for this split.")


def main() -> None:
    """CLI entrypoint for manifest preparation.

    Returns:
        ``None``.

    Logic:
        Parse CLI arguments, validate important optional combinations, and
        execute single-split manifest creation.
    """
    parser = argparse.ArgumentParser(
        description="Prepare a single, multimodal CT-RATE Benchmark manifest file."
    )

    # 1) Input file arguments.
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
    parser.add_argument(
        "--metadata_csv",
        type=str,
        default=None,
        help="(Optional) Metadata CSV with reconstruction IDs to expand labels.",
    )
    
    # 2) Output file argument.
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path for the final output manifest (e.g., 'manifests/train.csv').",
    )
    
    # 3) Visual-path generation arguments.
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
    parser.add_argument(
        "--visual_suffix",
        type=str,
        default=".pt",
        help="File suffix/extension to use for visual features (e.g., '.pt' or '.npz').",
    )
    
    # 4) Text-path generation arguments.
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
    parser.add_argument(
        "--text_suffix",
        type=str,
        default=".pt",
        help="File suffix/extension to use for text features (e.g., '.pt' or '.npz').",
    )

    # 5) Parse args and emit compatibility warnings.
    args = parser.parse_args()

    if args.reports_csv is None:
        log.warning("--- No --reports_csv provided. "
                    "Manifest will be VISUAL-ONLY and will not contain report text. ---")
    elif args.text_prefix is None:
        log.warning("--- No --text_prefix provided. "
                    "Manifest will include report text but NO text *feature path*. ---")

    # 6) Create manifest.
    create_manifest(args)


if __name__ == "__main__":
    main()