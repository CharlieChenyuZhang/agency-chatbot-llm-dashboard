#!/usr/bin/env python3
"""
Utility script to stitch together per-layer loss plots into a single grid image
for each behavioral trait.

Example:
    python combine_loss_plots.py --output-dir output/20251111_115518
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TRAITS = ("rigidity", "independence", "goal_persistence")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine per-layer loss PNGs into summary grids."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the timestamped output directory (e.g. output/20251111_115518).",
    )
    parser.add_argument(
        "--traits",
        nargs="*",
        default=TRAITS,
        help="Behavioral traits to combine. Defaults to all supported traits.",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=6,
        help="Number of columns in the grid. Adjust based on how many layers you trained.",
    )
    return parser


def combine_trait_plots(trait: str, directory: Path, columns: int) -> None:
    pattern = f"loss_curve_{trait}_*_layer_*.png"
    files = sorted(directory.glob(pattern), key=lambda p: p.stem)

    if not files:
        print(f"[WARN] No loss curve PNGs found for trait '{trait}' in {directory}")
        return

    total = len(files)
    columns = max(1, columns)
    rows = math.ceil(total / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 3, rows * 3))
    axes = np.atleast_2d(axes)

    # Flatten for easy indexing; extra axes stay blank
    flat_axes = axes.flatten()
    for ax in flat_axes:
        ax.axis("off")

    for idx, image_path in enumerate(files):
        ax = flat_axes[idx]
        img = plt.imread(image_path)
        ax.imshow(img)
        layer_label = image_path.stem.split("_layer_")[-1]
        ax.set_title(f"Layer {layer_label}", fontsize=10)

    fig.suptitle(f"{trait.replace('_', ' ').title()} Loss Curves", fontsize=14, weight="bold")
    plt.tight_layout()

    output_path = directory / f"{trait}_loss_grid.png"
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"[INFO] Saved combined plot for '{trait}' -> {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    output_dir = args.output_dir

    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")

    for trait in args.traits:
        combine_trait_plots(trait, output_dir, args.columns)


if __name__ == "__main__":
    main()

