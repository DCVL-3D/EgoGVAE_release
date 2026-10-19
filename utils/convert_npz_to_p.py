"""Merge EgoGVAE evaluation outputs into a single visualization file."""

import argparse
import re
from pathlib import Path

import joblib
import numpy as np


def natural_key(name: str):
    """Sort numeric filename segments numerically (e.g., 2 before 10)."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def extract_sample_id(path: Path) -> int:
    """Use the last number in a filename as its sample identifier."""
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"No numeric sample ID in filename: {path.name}")
    return int(numbers[-1])


def merge_all_npz_to_one_p(input_dir: Path, output_path: Path) -> None:
    """Save {sample_id: {array_name: array}} from all input NPZ files."""
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    npz_files = sorted(input_dir.glob("*.npz"), key=lambda p: natural_key(p.name))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    merged_data = {}
    for npz_path in npz_files:
        sample_id = extract_sample_id(npz_path)
        if sample_id in merged_data:
            raise ValueError(
                f"Duplicate sample ID {sample_id} from {npz_path.name}; "
                "please rename the files to give every sample a unique ID."
            )

        # Materialize the arrays before closing the NPZ archive.
        with np.load(npz_path, allow_pickle=True) as npz_data:
            merged_data[sample_id] = {
                name: npz_data[name] for name in npz_data.files
            }
        print(f"Loaded {npz_path.name} as sample {sample_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(merged_data, output_path)
    print(f"Saved {len(merged_data)} samples to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-sample EgoGVAE NPZ files for Viser visualization."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=Path("./outputs/results"),
        help="Directory containing per-sample .npz files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./outputs/merged_results"),
        help="Directory in which merged_results.p will be saved.",
    )
    args = parser.parse_args()
    merge_all_npz_to_one_p(args.input_dir, args.output_dir / "merged_results.p")


if __name__ == "__main__":
    main()
