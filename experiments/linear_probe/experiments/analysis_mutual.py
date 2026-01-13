"""
Plot mutual_predict metrics: macro F1 and exact match vs. layer for each dataset variant.

- Match the SAME color style / format as your example figure
- Align Macro-F1 y-axis range/ticks with the example figure
- Do NOT plot shuffle baseline (ignore datasets like *_shuffle)

Usage:
python -m experiments.linear_probe.experiments.analysis_mutual \
  --metrics_json path/to/mutual_metrics.json \
  --output_dir plots \
  --prefix qwen7b_mutual \
  --dpi 300
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# -----------------------------
# Style helpers (match your figure)
# -----------------------------
def rename(name: str) -> str:
    mapping = {
        "normal": "Normal Words",
        "perturbed": "Typo Words",
        "random": "Random Letters",
        "special": "Special Symbols",
    }
    return mapping.get(name, name)


def adjust_color_hsv(
    hex_color: str,
    saturation_factor: float = 1.0,
    value_factor: float = 1.0,
    hue_shift: float = 0.0,
) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0

    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + hue_shift) % 1.0
    s = max(0.0, min(1.0, s * saturation_factor))
    v = max(0.0, min(1.0, v * value_factor))

    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _is_shuffle_baseline(dataset_name: str) -> bool:
    # Mutual metrics often include: normal_shuffle / perturbed_shuffle / ...
    return dataset_name.endswith("_shuffle") or "shuffle" in dataset_name.lower()


# -----------------------------
# Data loading
# -----------------------------
def load_entries(metrics_path: Path) -> List[Dict]:
    data = json.loads(metrics_path.read_text())
    results = data.get("results", {})
    entries: List[Dict] = []

    for dataset_name, rows in results.items():
        # Skip shuffle baseline curves
        if _is_shuffle_baseline(dataset_name):
            continue

        for row in rows:
            layer = row.get("layer")
            if layer is None:
                continue
            entries.append(
                {
                    "dataset": dataset_name,
                    "layer": int(layer),
                    "em": row.get("test_em"),
                    "mf1": row.get("test_mf1"),
                }
            )

    if not entries:
        raise ValueError(f"No (non-shuffle) layer entries found in {metrics_path}")
    return entries


def _ordered_datasets(entries: List[Dict]) -> List[str]:
    preferred = ["normal", "perturbed", "random", "special"]
    present = {e["dataset"] for e in entries}

    ordered = [d for d in preferred if d in present]
    # append any unexpected dataset names (rare)
    for d in sorted(present):
        if d not in preferred:
            ordered.append(d)
    return ordered


# -----------------------------
# Plotting (match your figure)
# -----------------------------
def plot_metric(
    entries: List[Dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[float, float] = (12.0, 7.0),  # big figure by default
) -> None:
    datasets = _ordered_datasets(entries)

    # Palette + saturation (same as your plotting code)
    base_colors = ["#D99179", "#A4A1D9", "#98CEBA", "#DFC79F"]  # normal, typo, random, special
    colors = [adjust_color_hsv(c, saturation_factor=1.4, value_factor=1.0) for c in base_colors]
    color_map = {ds: colors[i % len(colors)] for i, ds in enumerate(["normal", "perturbed", "random", "special"])}

    # Scale sizes so that "big figsize" keeps the same visual weight as your small figure
    base_figsize = (5.5, 3.5)
    scale = min(figsize[0] / base_figsize[0], figsize[1] / base_figsize[1])

    xlabel_fs = 15 * scale
    ylabel_fs = 15 * scale
    tick_fs = 12 * scale
    legend_fs = 9 * scale
    lw = 2 * scale
    ms = 5 * scale
    grid_lw = 0.25 * scale

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    max_layer = 0
    for ds in datasets:
        ds_entries = [e for e in entries if e["dataset"] == ds and e.get(metric_key) is not None]
        if not ds_entries:
            continue
        ds_entries.sort(key=lambda x: x["layer"])
        layers = [int(e["layer"]) for e in ds_entries]
        values = [float(e[metric_key]) for e in ds_entries]  # type: ignore[arg-type]
        max_layer = max(max_layer, max(layers))

        ax.plot(
            layers,
            values,
            marker="o",
            markersize=ms,
            color=color_map.get(ds, colors[0]),
            linewidth=lw,
            linestyle="-",
            alpha=1.0,
            label=rename(ds),
        )

    # Axes labels / ticks
    ax.set_xlabel("Layer", fontsize=xlabel_fs)
    ax.set_ylabel(ylabel, fontsize=ylabel_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)

    # X ticks like your figure: 0,5,10,...
    if max_layer > 0:
        ax.set_xticks(list(range(0, max_layer + 1, 5)))

    # Y axis alignment with your example figure (Macro F1)
    if metric_key == "mf1":
        ax.set_ylim(0.13, 0.565)
        ax.set_yticks([0.15 + 0.05 * i for i in range(9)])  # 0.15..0.55

    # Spines / grid (same as your figure)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls="--", lw=grid_lw, color="#4E616C")

    # Legend format (same as your figure; no baseline entries, still 2 columns)
    ax.legend(
        fontsize=legend_fs,
        bbox_to_anchor=(1.00, 0.13),
        loc="lower right",
        ncol=2,
        frameon=True,
        framealpha=0.95,
        edgecolor="#CCCCCC",
        borderpad=0.5 * scale,
        labelspacing=0.4 * scale,
        columnspacing=1.2 * scale,
    )

    # Title is optional; your example figure doesn't show title (keep it off by default)
    # If you want it, uncomment:
    # ax.set_title(title, fontsize=12 * scale)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi, format="png")
    plt.close(fig)
    print(f"Saved {output_path}")


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot mutual_predict metrics (macro F1 and exact match) vs. layer.")
    parser.add_argument("--metrics_json", type=Path, required=True, help="Path to mutual_predict metrics.json.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help="Directory to save plots. Default: same directory as metrics_json.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Filename prefix for plots. Default: metrics_json stem.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output PNGs (higher = sharper).")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 7.0),
        metavar=("W", "H"),
        help="Figure size in inches (big figure by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics_json
    output_dir = args.output_dir or metrics_path.parent
    prefix = args.prefix or metrics_path.stem
    figsize = (float(args.figsize[0]), float(args.figsize[1]))

    entries = load_entries(metrics_path)

    f1_path = output_dir / f"{prefix}_mf1.png"
    em_path = output_dir / f"{prefix}_em.png"

    plot_metric(
        entries,
        metric_key="mf1",
        ylabel="Test Macro F1",
        title="Test Macro F1 vs. Layer",
        output_path=f1_path,
        dpi=args.dpi,
        figsize=figsize,
    )
    plot_metric(
        entries,
        metric_key="em",
        ylabel="Test Exact Match",
        title="Test Exact Match vs. Layer",
        output_path=em_path,
        dpi=args.dpi,
        figsize=figsize,
    )

    print(f"Saved plots:\n- {f1_path}\n- {em_path}")


if __name__ == "__main__":
    main()
