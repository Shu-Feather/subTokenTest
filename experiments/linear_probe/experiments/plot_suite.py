"""
Plot test EM and test Macro F1 across layers for multiple experiments.

Assumes a directory structure like:
  base_dir/
    normal/metrics.json
    normal_shuffle/metrics.json
    perturbed/metrics.json
    ...

Each metrics.json is produced by number_probe.py and contains per-layer results.
Outputs two PNGs under base_dir:
  - test_em_vs_layer.png
  - test_macro_f1_vs_layer.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

def rename(name: str) -> str:
    """Rename experiment names for better plot legends."""
    mapping = {
        "normal": "Normal Words",
        "perturbed": "Typo Words",
        "random": "Random Letters",
        "special": "Special Symbols",
        "normal_shuffle": "Normal Words Baseline",
        "perturbed_shuffle": "Typo Words Baseline",
        "random_shuffle": "Random Letters Baseline",
        "special_shuffle": "Special Symbols Baseline"
    }
    return mapping.get(name, name)

import colorsys
from typing import Tuple, Union

def adjust_color_hsv(
    hex_color: str,
    saturation_factor: float = 1.0,
    value_factor: float = 1.0,
    hue_shift: float = 0.0
) -> str:
    """
    使用HSV颜色空间调整颜色
    
    Args:
        hex_color: 十六进制颜色字符串，如 "#D99179"
        saturation_factor: 饱和度乘数 (>1增加饱和度，<1降低饱和度)
        value_factor: 亮度乘数 (>1变亮，<1变暗)
        hue_shift: 色相偏移 (0-1之间，0.1表示顺时针旋转36度)
    
    Returns:
        调整后的十六进制颜色字符串
    """
    # 移除#号并转换为RGB
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    
    # 转换为HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    
    # 调整参数
    h = (h + hue_shift) % 1.0  # 色相循环
    s = max(0.0, min(1.0, s * saturation_factor))  # 限制在0-1之间
    v = max(0.0, min(1.0, v * value_factor))  # 限制在0-1之间
    
    # 转回RGB
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    
    # 转回十六进制
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"

def load_metrics_file(path: Path) -> Dict[int, Dict[str, float]]:
    data = json.loads(path.read_text())
    layers = {}
    for entry in data.get("results", []):
        layer = entry["layer"]
        layers[layer] = {
            "test_em": entry.get("test_em"),
            "test_mf1": entry.get("test_mf1") or entry.get("test_mf1".lower()),
        }
    return layers


def load_experiments(base_dir: Path) -> Dict[str, Dict[int, Dict[str, float]]]:
    experiments = {}
    for sub in base_dir.iterdir():
        if not sub.is_dir():
            continue
        metrics_path = sub / "metrics.json"
        if metrics_path.exists():
            experiments[sub.name] = load_metrics_file(metrics_path)
    return experiments


def plot_metric(
    experiments: Dict[str, Dict[int, Dict[str, float]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int = 300,
) -> None:
    colors = ["#D99179", "#A4A1D9", "#98CEBA", "#DFC79F"]
    colors = [adjust_color_hsv(c, saturation_factor=1.4, value_factor=1.0,) for c in colors]
    
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))
    
    # 定义实验顺序（包括 shuffle 版本）
    experiment_order = [
        "normal", "normal_shuffle",
        "perturbed", "perturbed_shuffle", 
        "random", "random_shuffle",
        "special", "special_shuffle"
    ]
    
    color_idx = 0
    for name in experiment_order:
        if name not in experiments:
            continue
            
        layers = experiments[name]
        xs: List[int] = []
        ys: List[float] = []
        
        for layer, vals in sorted(layers.items()):
            val = vals.get(metric_key)
            if val is None:
                continue
            xs.append(layer)
            ys.append(val)
        
        if xs and ys:
            is_shuffle = name.endswith("_shuffle")
            linestyle = "--" if is_shuffle else "-"
            alpha = 0.7 if is_shuffle else 1.0
            
            # baseline and main experiment share the same color
            current_color = colors[(color_idx // 2) % len(colors)]
            
            ax.plot(xs, ys, 
                    marker="o" if not is_shuffle else "s",  # shuffle is square marker
                    markersize=5 if not is_shuffle else 4,
                    color=current_color,
                    linewidth=2,
                    linestyle=linestyle,
                    alpha=alpha,
                    label=rename(name))
            color_idx += 1
    
    plt.xlabel("Layer", fontsize=15)
    plt.ylabel(ylabel, fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(ls="--", lw=0.25, color="#4E616C")
    
    plt.legend(
        fontsize=9,
        bbox_to_anchor=(1.00, 0.13),  # 右下角位置
        loc='lower right',
        ncol=2,                        # 2列
        frameon=True,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        borderpad=0.5,
        labelspacing=0.4,
        columnspacing=1.2
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    print(f"Saved {output_path}")

import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List

def plot_metric_plotly(
    experiments: Dict[str, Dict[int, Dict[str, float]]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    dpi: int = 300,
) -> None:
    # 创建图形
    fig = go.Figure()
    
    colors = ["#D99179", "#A4A1D9", "#98CEBA", "#DFC79F"]
    color_idx = 0
    
    # 为每个实验添加轨迹
    for name, layers in experiments.items():
        xs: List[int] = []
        ys: List[float] = []
        
        for layer, vals in sorted(layers.items()):
            val = vals.get(metric_key)
            if val is None:
                continue
            xs.append(layer)
            ys.append(val)
        
        if xs and ys:
            if not name.endswith("_shuffle"):
                # 添加轨迹
                current_color = colors[color_idx % len(colors)]
                color_idx += 1
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='lines+markers',
                    marker=dict(
                        size=6,
                        symbol='circle',
                        color=current_color,
                    ),
                    line=dict(
                        width=2,
                        color=current_color,
                    ),
                    name=rename(name)
                ))
    
    # 更新布局
    fig.update_layout(
        xaxis=dict(
            title=dict(
                text="Layer",
                font=dict(size=15)
            ),
        ),
        yaxis=dict(
            title=dict(
                text=ylabel,
                font=dict(size=15)
            ),
        ),
        legend=dict(
            font=dict(size=12),
            x=0.65,
            y=0.04,
            bgcolor="rgba(255, 255, 255, 0.8)",  # 半透明白色背景
        ),
        width=480,
        height=300,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    
    # # 隐藏右轴和上轴的边框（通过设置颜色为透明）
    # fig.update_layout(
    #     xaxis=dict(
    #         **fig.layout.xaxis.to_plotly_json(),
    #         showspikes=False
    #     ),
    #     yaxis=dict(
    #         **fig.layout.yaxis.to_plotly_json(),
    #         showspikes=False
    #     )
    # )
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存为静态图像（PNG, JPEG, SVG等）
    fig.write_image(str(output_path), scale=dpi/96)  # plotly的scale基于96dpi
    
    print(f"Saved {output_path}")
    
    
def main() -> None:
    parser = argparse.ArgumentParser(description="Plot test EM and Macro F1 across experiments.")
    parser.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="Directory containing subfolders with metrics.json (e.g., normal, normal_shuffle...).",
    )
    args = parser.parse_args()

    experiments = load_experiments(args.base_dir)
    if not experiments:
        raise FileNotFoundError(f"No metrics.json found under {args.base_dir}")

    plot_metric(
        experiments,
        metric_key="test_em",
        ylabel="Test Exact Match",
        title="Test EM vs Layer",
        output_path=args.base_dir / "test_em_vs_layer_hy.pdf",
    )
    plot_metric(
        experiments,
        metric_key="test_mf1",
        ylabel="Test Macro F1",
        title="Test Macro F1 vs Layer",
        output_path=args.base_dir / "test_macro_f1_vs_layer_hy.pdf",
    )


if __name__ == "__main__":
    main()
