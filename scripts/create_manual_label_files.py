"""Create manual label files (train.csv, val.csv) using AccessionNo mappings.

This script uses:
- data/labels/manual_labels/all_manual_labels.csv
- data/labels/manual_labels/map_train_classifier.csv
- data/labels/manual_labels/map_val_classifier.csv

It writes:
- data/labels/manual_labels/train.csv
- data/labels/manual_labels/val.csv

Logic:
1) Filter mapping rows where NameinCTRATE == 'not in CT-RATE'.
2) Match mappings to all_manual_labels.csv by AccessionNo.
3) Replace AccessionNo with VolumeName (from NameinCTRATE).
4) Save train/val label files.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


NOT_IN_CT_RATE = "not in CT-RATE"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def _clean_labels_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    if "AccessionNo" not in df.columns:
        raise ValueError(f"Missing 'AccessionNo' column in {path}")
    out = df.copy()
    out["AccessionNo"] = out["AccessionNo"].astype(str).str.strip()
    return out


def _clean_map_df(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    required = {"AccessionNo", "NameinCTRATE"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")

    out = df.copy()
    out["AccessionNo"] = out["AccessionNo"].astype(str).str.strip()
    out["NameinCTRATE"] = out["NameinCTRATE"].astype(str).str.strip()
    out = out[out["NameinCTRATE"].str.lower() != NOT_IN_CT_RATE.lower()].copy()
    return out


def _build_label_file(labels_df: pd.DataFrame, map_df: pd.DataFrame) -> pd.DataFrame:
    merged = labels_df.merge(map_df[["AccessionNo", "NameinCTRATE"]], on="AccessionNo", how="inner")

    merged = merged.drop(columns=["AccessionNo"])
    merged = merged.rename(columns={"NameinCTRATE": "VolumeName"})

    cols = ["VolumeName"] + [col for col in merged.columns if col != "VolumeName"]
    merged = merged[cols]

    merged = merged.drop_duplicates(subset=["VolumeName"]).sort_values("VolumeName").reset_index(drop=True)
    return merged


def create_manual_label_files(
    all_labels_csv: Path,
    map_train_csv: Path,
    map_val_csv: Path,
    output_dir: Path,
) -> None:
    labels_df = _clean_labels_df(_read_csv(all_labels_csv), all_labels_csv)
    map_train_df = _clean_map_df(_read_csv(map_train_csv), map_train_csv)
    map_val_df = _clean_map_df(_read_csv(map_val_csv), map_val_csv)

    train_accessions = set(map_train_df["AccessionNo"])
    val_accessions = set(map_val_df["AccessionNo"])
    overlap = train_accessions.intersection(val_accessions)
    if overlap:
        raise ValueError(
            "Accession overlap found between train/val mapping files: "
            f"{len(overlap)} overlapping IDs."
        )

    train_df = _build_label_file(labels_df, map_train_df)
    val_df = _build_label_file(labels_df, map_val_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    train_out = output_dir / "train.csv"
    val_out = output_dir / "val.csv"

    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    log.info("Wrote: %s (%s rows)", train_out, len(train_df))
    log.info("Wrote: %s (%s rows)", val_out, len(val_df))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create manual train/val label files from accession mappings."
    )
    parser.add_argument(
        "--all-labels-csv",
        type=Path,
        default=Path("data/labels/manual_labels/all_manual_labels.csv"),
        help="Path to all manual labels CSV.",
    )
    parser.add_argument(
        "--map-train-csv",
        type=Path,
        default=Path("data/labels/manual_labels/map_train_classifier.csv"),
        help="Path to train mapping CSV.",
    )
    parser.add_argument(
        "--map-val-csv",
        type=Path,
        default=Path("data/labels/manual_labels/map_val_classifier.csv"),
        help="Path to val mapping CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/labels/manual_labels"),
        help="Directory for output train.csv and val.csv.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    create_manual_label_files(
        all_labels_csv=args.all_labels_csv,
        map_train_csv=args.map_train_csv,
        map_val_csv=args.map_val_csv,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
