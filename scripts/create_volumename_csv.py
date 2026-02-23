"""Generate a manifest-style CSV from extracted feature filenames.

The script scans a feature directory, derives volume identifiers from
file stems, appends a configurable suffix, and writes a one-column CSV.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd


logging.basicConfig(
	level=logging.INFO,
	format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for CSV generation.

	Returns:
		Parsed ``argparse.Namespace`` with input/output and naming options.

	Logic:
		Define a small CLI surface that controls file discovery and output
		format without hard-coded paths.
	"""
	parser = argparse.ArgumentParser(
		description="Create a CSV listing volume identifiers derived from feature filenames."
	)
	parser.add_argument(
		"--feature-dir",
		type=Path,
		required=True,
		help="Directory containing extracted feature files.",
	)
	parser.add_argument(
		"--output-csv",
		type=Path,
		required=True,
		help="Output CSV path.",
	)
	parser.add_argument(
		"--glob-pattern",
		type=str,
		default="*.npz",
		help="Glob pattern used to discover feature files (default: '*.npz').",
	)
	parser.add_argument(
		"--column-name",
		type=str,
		default="VolumeName",
		help="Output column name (default: 'VolumeName').",
	)
	parser.add_argument(
		"--append-suffix",
		type=str,
		default=".nii.gz",
		help="Suffix appended to each file stem (default: '.nii.gz').",
	)
	return parser.parse_args()


def create_volumename_csv(
	feature_dir: Path,
	output_csv: Path,
	glob_pattern: str,
	column_name: str,
	append_suffix: str,
) -> int:
	"""Create a one-column CSV of volume names from feature files.

	Args:
		feature_dir: Directory containing feature files.
		output_csv: Destination CSV file path.
		glob_pattern: File pattern used to discover features.
		column_name: Name of the output CSV column.
		append_suffix: String appended to each discovered file stem.

	Returns:
		Number of rows written to ``output_csv``.

	Logic:
		Validate the input directory, discover matching files in deterministic
		order, transform stems into volume identifiers, and persist them as CSV.
	"""
	# 1) Validate input directory.
	if not feature_dir.exists() or not feature_dir.is_dir():
		raise FileNotFoundError(f"Feature directory not found: {feature_dir}")

	# 2) Discover feature files and derive volume names.
	feature_files = sorted(p for p in feature_dir.glob(glob_pattern) if p.is_file())
	volume_names = [f"{path.stem}{append_suffix}" for path in feature_files]

	# 3) Write output CSV.
	df = pd.DataFrame({column_name: volume_names})

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	df.to_csv(output_csv, index=False)

	return len(df)


def main() -> None:
	"""CLI entrypoint for volume-name CSV generation.

	Returns:
		``None``.

	Logic:
		Parse CLI inputs, run CSV generation, and log a concise completion
		summary.
	"""
	# 1) Parse CLI arguments.
	args = parse_args()

	# 2) Execute generation.
	log.info("Scanning feature directory: %s", args.feature_dir)
	row_count = create_volumename_csv(
		feature_dir=args.feature_dir,
		output_csv=args.output_csv,
		glob_pattern=args.glob_pattern,
		column_name=args.column_name,
		append_suffix=args.append_suffix,
	)

	# 3) Log summary.
	log.info("Wrote %s rows to %s", row_count, args.output_csv)


if __name__ == "__main__":
	main()