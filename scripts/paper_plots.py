#!/usr/bin/env python3
"""
Generate paper-quality plots from pre-aggregated CSV artifacts.

This script reads each root directory's `runs_flat.csv` and optional
`token_usage.csv`, then produces publication figures. All plots are written
to a configurable --outdir.

Plots:
  1. Bar chart: control-flow pattern prevalence (rerouting/looping/termination) by file type
  2. Boxplot: normalized divergence step (t*/T) by file type (4-5 representative types)
  3. Heatmap: perturbation type x control-flow pattern (%), no grid
  4. Boxplot: extra tokens by file type
  4b. Boxplot: extra tokens by perturbation type
  5. Grouped bar chart: LLM x control-flow pattern (prevalence %)
  6. Boxplot: normalized first divergence position by perturbation type
  7. Boxplot: normalized edit distance by perturbation type
  8. First divergence position by LLM (single plot, one box per model; y = LLM, x = position)
Plots 1–4, 6–7 use combined data from all given roots. Plots 5 and 8 use per-model data from the same roots.

Usage:
  python paper_plots.py [--roots ROOT [ROOT ...]] [--outdir DIR]

    Default --roots: data/analysis_val_gpt, data/analysis_val_llama, data/analysis_val_qwen.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Ensure fonts are embedded as TrueType (required for camera-ready submissions)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Colorblind-friendly palette (IBM Design)
COLORS = {
    "blue": "#648FFF",
    "purple": "#785EF0",
    "magenta": "#DC267F",
    "orange": "#FE6100",
    "yellow": "#FFB000",
    "gray": "#808080",
    "light_gray": "#D3D3D3",
}

# File type colors
FILE_TYPE_COLORS = {
    "tabular": "#648FFF",
    "document": "#785EF0",
    "image": "#DC267F",
    "audio": "#FE6100",
    "any": "#808080",
    "unknown": "#D3D3D3",
    "data": "#FFB000",
}

# Canonical order for file type and manifestation type (same order across all plots)
FILE_TYPE_ORDER = ["tabular", "document", "image", "audio", "any", "unknown", "data"]
MANIFESTATION_TYPE_ORDER = [
    "catastrophic_failure",
    "early_termination",
    "loop_or_extended_execution",
    "silent_semantic_corruption",
    "strategy_reroute",
    "structural_divergence_with_outcome_change",
    "structural_divergence_recovered",
    "no_observable_effect",
    "outcome_change_uncategorized",
]


def _file_type_order_subset(present: List[str], top_n: Optional[int] = None) -> List[str]:
    """Return present file types in canonical order, optionally limited to first top_n."""
    ordered = [ft for ft in FILE_TYPE_ORDER if ft in present]
    if top_n is not None:
        ordered = ordered[:top_n]
    return ordered


def _manifestation_order_subset(present: List[str]) -> List[str]:
    """Return present manifestation types in canonical order."""
    return [m for m in MANIFESTATION_TYPE_ORDER if m in present]


def _file_type_display(ft: str) -> str:
    """Display label for file type: 'unknown' -> 'slide deck', others unchanged."""
    return "slide deck" if ft == "unknown" else ft


DIVERGENCE_POSITION_LABEL = "First divergence position (normalized)"


def setup_paper_style() -> None:
    """Configure matplotlib for publication figures. Fallback if seaborn style missing."""
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except (OSError, KeyError):
        try:
            plt.style.use("seaborn-whitegrid")
        except (OSError, KeyError):
            plt.style.use("ggplot")
    import matplotlib as mpl
    mpl.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _sanitize_prefix(prefix: str) -> str:
    """Make a string safe for use as a filename prefix (no path separators or problematic chars)."""
    s = str(prefix).strip()
    for c in "/\\:*?\"<>|":
        s = s.replace(c, "_")
    return s or "unnamed"


def _infer_model_name(root: Path) -> str:
    """Infer model label from analysis directory name."""
    name = root.name.lower()
    if "analysis_val_" in name:
        suffix = name.split("analysis_val_", 1)[1]
        if suffix:
            return suffix
    if "gpt" in name:
        return "gpt"
    if "llama" in name:
        return "llama"
    if "qwen" in name:
        return "qwen"
    return root.name


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False})
        .fillna(False)
    )


# ---------------------------------------------------------------------------
# Plot 1: Control-flow pattern prevalence by file type
# ---------------------------------------------------------------------------

def plot_control_flow_by_file_type(df: pd.DataFrame, outdir: Path, file_prefix: Optional[str] = None) -> None:
    """Bar chart: rerouting / looping / termination prevalence by file type."""
    _ensure_outdir(outdir)
    base = "fig_control_flow_by_file_type"
    name = f"{_sanitize_prefix(file_prefix)}-{base}" if file_prefix else base
    if df.empty:
        return

    plot_df = df[df["file_type"].notna()].copy()
    if plot_df.empty:
        return

    plot_df["has_reroute"] = (plot_df["reroutes"].fillna(0) > 0) if "reroutes" in plot_df.columns else False
    plot_df["has_loop"] = plot_df["extended_execution"].fillna(False) if "extended_execution" in plot_df.columns else False
    plot_df["has_termination"] = plot_df["early_termination"].fillna(False) if "early_termination" in plot_df.columns else False

    agg = plot_df.groupby("file_type", dropna=False).agg(
        n=("task_id", "count"),
        pct_reroute=("has_reroute", "mean"),
        pct_loop=("has_loop", "mean"),
        pct_termination=("has_termination", "mean"),
    ).reset_index()
    agg["pct_reroute"] = 100.0 * agg["pct_reroute"]
    agg["pct_loop"] = 100.0 * agg["pct_loop"]
    agg["pct_termination"] = 100.0 * agg["pct_termination"]

    # Sort by canonical file type order (consistent across all plots)
    order = _file_type_order_subset(agg["file_type"].tolist())
    agg = agg.set_index("file_type").loc[order].reset_index()
    if agg.empty:
        return

    x = np.arange(len(agg))
    w = 0.25

    fig, ax = plt.subplots(figsize=(9, max(7, 0.7 * len(agg))))
    ax.bar(x - w, agg["pct_reroute"], width=w, label="Rerouting", color=COLORS["blue"], alpha=0.8)
    ax.bar(x, agg["pct_loop"], width=w, label="Looping (extended)", color=COLORS["orange"], alpha=0.8)
    ax.bar(x + w, agg["pct_termination"], width=w, label="Early termination", color=COLORS["purple"], alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels([_file_type_display(ft) for ft in agg["file_type"].tolist()], rotation=0, fontsize=16)
    ax.set_ylabel("Prevalence (%)", fontsize=16)
    ax.set_xlabel("File type", fontsize=16)
    ax.set_title("Control-flow pattern prevalence by file type", fontsize=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 105)

    fig.tight_layout(rect=[0, 0.10, 1, 1])
    fig.savefig(outdir / f"{name}.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2: Normalized divergence step (t*/T) by file type
# ---------------------------------------------------------------------------

def plot_divergence_step_by_file_type(df: pd.DataFrame, outdir: Path, top_n: int = 5, file_prefix: Optional[str] = None) -> None:
    """Boxplot: divergence_normalized_position by file type. Uses 'file type' in labels."""
    _ensure_outdir(outdir)
    base = "fig_divergence_step_by_file_type"
    name = f"{_sanitize_prefix(file_prefix)}-{base}" if file_prefix else base
    plot_df = df[df["divergence_normalized_position"].notna() & df["file_type"].notna()].copy()
    if plot_df.empty:
        return

    # Top file types by run count, then sort by canonical file type order
    present = plot_df["file_type"].value_counts().head(top_n).index.tolist()
    order = _file_type_order_subset(present)
    if not order:
        order = present
    plot_df = plot_df[plot_df["file_type"].isin(order)]

    if not order:
        return

    data = [plot_df.loc[plot_df["file_type"] == ft, "divergence_normalized_position"].values for ft in order]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(order))))
    line_color = "#333333"
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=[_file_type_display(ft) for ft in order],
        patch_artist=True,
        widths=0.6,
        flierprops={"markersize": 3, "color": line_color},
        boxprops=dict(edgecolor="none"),
        medianprops=dict(color=line_color, linewidth=1.5),
        whiskerprops=dict(color=line_color),
        capprops=dict(color=line_color),
    )
    for i, (patch, ft) in enumerate(zip(bp["boxes"], order)):
        patch.set_facecolor(FILE_TYPE_COLORS.get(ft, COLORS["gray"]))
        patch.set_alpha(0.7)

    ax.set_xlabel(DIVERGENCE_POSITION_LABEL, fontsize=16)
    ax.set_ylabel("File type", fontsize=16)
    ax.set_title("Normalized divergence step by file type", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(-0.02, 0.8)

    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3: Heatmap file type x manifestation type
# ---------------------------------------------------------------------------

def plot_heatmap_file_type_manifestation(df: pd.DataFrame, outdir: Path, file_prefix: Optional[str] = None) -> None:
    """Heatmap: rows = perturbation type, columns = control-flow pattern (%), no grid."""
    _ensure_outdir(outdir)
    base = "fig_heatmap_file_type_manifestation"
    name = f"{_sanitize_prefix(file_prefix)}-{base}" if file_prefix else base
    plot_df = df[df["perturbation_type"].notna()].copy()
    if plot_df.empty:
        return

    plot_df["has_reroute"] = (plot_df["reroutes"].fillna(0) > 0) if "reroutes" in plot_df.columns else False
    plot_df["has_loop"] = plot_df["extended_execution"].fillna(False) if "extended_execution" in plot_df.columns else False
    plot_df["has_termination"] = plot_df["early_termination"].fillna(False) if "early_termination" in plot_df.columns else False

    cf_cols = ["Rerouting", "Looping (extended)", "Early termination"]
    agg = plot_df.groupby("perturbation_type", dropna=False).agg(
        has_reroute=("has_reroute", "mean"),
        has_loop=("has_loop", "mean"),
        has_termination=("has_termination", "mean"),
    )
    agg.columns = cf_cols
    agg = agg * 100.0
    row_order = agg.sum(axis=1).sort_values(ascending=False).index.tolist()
    crosstab_pct = agg.reindex(index=row_order).fillna(0)
    if crosstab_pct.empty:
        return

    fig, ax = plt.subplots(figsize=(max(8, 0.4 * crosstab_pct.shape[1]), max(6, 0.35 * crosstab_pct.shape[0])))
    im = ax.imshow(crosstab_pct.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(np.arange(crosstab_pct.shape[1]))
    ax.set_yticks(np.arange(crosstab_pct.shape[0]))
    ax.set_xticklabels(crosstab_pct.columns.tolist(), rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels([c.replace("_", " ") for c in crosstab_pct.index.tolist()], fontsize=16)
    ax.set_xlabel("Control-flow pattern", fontsize=17)
    ax.set_ylabel("Perturbation type", fontsize=17)
    ax.set_title("Perturbation type × control-flow pattern (%)", fontsize=18)
    ax.grid(False)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Prevalence (%)", fontsize=16)
    cbar.ax.tick_params(labelsize=16)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4: Extra tokens by file type
# ---------------------------------------------------------------------------

def plot_extra_tokens_by_file_type(df: pd.DataFrame, outdir: Path, file_prefix: Optional[str] = None) -> None:
    """Boxplot: extra tokens (perturbed - baseline) by file type. Only runs with extra_tokens >= 0. X-axis limit 0 to 50 (thousands)."""
    _ensure_outdir(outdir)
    base = "fig_extra_tokens_by_file_type"
    name = f"{_sanitize_prefix(file_prefix)}-{base}" if file_prefix else base
    plot_df = df[df["extra_tokens"].notna() & df["file_type"].notna() & (df["extra_tokens"] >= 0)].copy()
    if plot_df.empty:
        return

    order = plot_df.groupby("file_type")["extra_tokens"].median().sort_values(ascending=False).index.tolist()
    order = _file_type_order_subset(order)
    if not order:
        order = plot_df["file_type"].unique().tolist()
    data = [plot_df.loc[plot_df["file_type"] == ft, "extra_tokens"].values for ft in order]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(order))))
    line_color = "#333333"
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=[_file_type_display(ft) for ft in order],
        patch_artist=True,
        widths=0.6,
        flierprops={"markersize": 3, "color": line_color},
        boxprops=dict(edgecolor="none"),
        medianprops=dict(color=line_color, linewidth=1.5),
        whiskerprops=dict(color=line_color),
        capprops=dict(color=line_color),
    )
    for patch, ft in zip(bp["boxes"], order):
        patch.set_facecolor(FILE_TYPE_COLORS.get(ft, COLORS["gray"]))
        patch.set_alpha(0.7)
    ax.set_xlabel("Extra tokens", fontsize=16)
    ax.set_ylabel("File type", fontsize=16)
    ax.set_title("Extra tokens by file type", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(-500, 50_000)  # Slightly below 0 so values at 0 show; 0 to 50k (axis may show as 0–50 with ×10³)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4b: Extra tokens by perturbation type
# ---------------------------------------------------------------------------

def plot_extra_tokens_by_perturbation(df: pd.DataFrame, outdir: Path, top_n: int = 18, file_prefix: Optional[str] = None) -> None:
    """Boxplot: extra tokens (perturbed - baseline) by perturbation type. Only runs with extra_tokens >= 0."""
    _ensure_outdir(outdir)
    base = "fig_extra_tokens_by_perturbation"
    name = f"{_sanitize_prefix(file_prefix)}-{base}" if file_prefix else base
    plot_df = df[df["extra_tokens"].notna() & df["perturbation_type"].notna() & (df["extra_tokens"] >= 0)].copy()
    if plot_df.empty:
        return

    order = (
        plot_df.groupby("perturbation_type")["extra_tokens"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    if len(order) > top_n:
        order = order[:top_n]
        plot_df = plot_df[plot_df["perturbation_type"].isin(order)]
    if not order:
        return

    data = [plot_df.loc[plot_df["perturbation_type"] == p, "extra_tokens"].values for p in order]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.4 * len(order))))
    line_color = "#333333"
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=[p.replace("_", " ") for p in order],
        patch_artist=True,
        widths=0.6,
        flierprops={"markersize": 3, "color": line_color},
        boxprops=dict(edgecolor="none"),
        medianprops=dict(color=line_color, linewidth=1.5),
        whiskerprops=dict(color=line_color),
        capprops=dict(color=line_color),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["magenta"])
        patch.set_alpha(0.7)
    ax.set_xlabel("Extra tokens", fontsize=16)
    ax.set_ylabel("Perturbation type", fontsize=16)
    ax.set_title("Extra tokens by perturbation type", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(-500, 50_000)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(3, 3))
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5: Grouped bar chart LLM x control-flow pattern (prevalence %)
# ---------------------------------------------------------------------------

def plot_llm_control_flow_prevalence(df: pd.DataFrame, outdir: Path) -> None:
    """Grouped bar chart: X = model (LLM), grouped bars = control-flow pattern (reroute/loop/termination), Y = prevalence %."""
    _ensure_outdir(outdir)
    plot_df = df[df["model"].notna()].copy()
    if plot_df.empty:
        return

    plot_df["has_reroute"] = (plot_df["reroutes"].fillna(0) > 0) if "reroutes" in plot_df.columns else False
    plot_df["has_loop"] = plot_df["extended_execution"].fillna(False) if "extended_execution" in plot_df.columns else False
    plot_df["has_termination"] = plot_df["early_termination"].fillna(False) if "early_termination" in plot_df.columns else False

    models = plot_df["model"].unique().tolist()
    patterns = [
        ("has_reroute", "Rerouting", COLORS["blue"]),
        ("has_loop", "Looping (extended)", COLORS["orange"]),
        ("has_termination", "Early termination", COLORS["purple"]),
    ]

    n_models = len(models)
    n_patterns = len(patterns)
    width = 0.8 / max(n_patterns, 1)
    x = np.arange(n_models)
    # Match control-flow-by-file-type dimensions: width 9, height scales like that plot
    fig, ax = plt.subplots(figsize=(9, max(7, 0.7 * n_models)))

    for i, (col, label, color) in enumerate(patterns):
        pcts = []
        for mod in models:
            sub = plot_df[plot_df["model"] == mod]
            total = len(sub)
            count = sub[col].sum()
            pcts.append(100.0 * count / total if total else 0)
        offset = (i - n_patterns / 2 + 0.5) * width
        ax.bar(x + offset, pcts, width, label=label, color=color, alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0, fontsize=16)
    ax.set_ylabel("Prevalence (%)", fontsize=16)
    ax.set_xlabel("Model (LLM)", fontsize=16)
    ax.set_title("Control-flow pattern prevalence by LLM", fontsize=18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=16)
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylim(0, 105)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(outdir / "fig_llm_control_flow_prevalence.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 6: Normalized first divergence position by perturbation type
# ---------------------------------------------------------------------------

def plot_divergence_step_by_perturbation(df: pd.DataFrame, outdir: Path, top_n: int = 18, file_prefix: Optional[str] = None) -> None:
    """Boxplot: divergence_normalized_position by perturbation type (horizontal, top_n by count)."""
    _ensure_outdir(outdir)
    base = "fig_divergence_step_by_perturbation"
    name = f"{_sanitize_prefix(file_prefix)}-{base}" if file_prefix else base
    plot_df = df[df["divergence_normalized_position"].notna() & df["perturbation_type"].notna()].copy()
    if plot_df.empty:
        return

    freq = plot_df["perturbation_type"].value_counts()
    keep = freq.head(top_n).index.tolist()
    plot_df = plot_df[plot_df["perturbation_type"].isin(keep)]
    order = (
        plot_df.groupby("perturbation_type")["divergence_normalized_position"]
        .median()
        .sort_values()
        .index.tolist()
    )
    if not order:
        return

    data = [plot_df.loc[plot_df["perturbation_type"] == p, "divergence_normalized_position"].values for p in order]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.4 * len(order))))
    line_color = "#333333"
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=[p.replace("_", " ") for p in order],
        patch_artist=True,
        widths=0.6,
        flierprops={"markersize": 3, "color": line_color},
        boxprops=dict(edgecolor="none"),
        medianprops=dict(color=line_color, linewidth=1.5),
        whiskerprops=dict(color=line_color),
        capprops=dict(color=line_color),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["purple"])
        patch.set_alpha(0.7)

    ax.set_xlabel(DIVERGENCE_POSITION_LABEL, fontsize=16)
    ax.set_ylabel("Perturbation type", fontsize=16)
    ax.set_title("Normalized first divergence position by perturbation type", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(-0.02, 0.8)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 7: Normalized edit distance by perturbation type
# ---------------------------------------------------------------------------

def plot_edit_distance_by_perturbation(df: pd.DataFrame, outdir: Path, top_n: int = 18, file_prefix: Optional[str] = None) -> None:
    """Boxplot: edit_distance_normalized by perturbation type (horizontal, top_n by count)."""
    _ensure_outdir(outdir)
    base = "fig_edit_distance_by_perturbation"
    name = f"{_sanitize_prefix(file_prefix)}-{base}" if file_prefix else base
    plot_df = df[df["edit_distance_normalized"].notna() & df["perturbation_type"].notna()].copy()
    if plot_df.empty:
        return

    freq = plot_df["perturbation_type"].value_counts()
    keep = freq.head(top_n).index.tolist()
    plot_df = plot_df[plot_df["perturbation_type"].isin(keep)]
    order = (
        plot_df.groupby("perturbation_type")["edit_distance_normalized"]
        .median()
        .sort_values()
        .index.tolist()
    )
    if not order:
        return

    data = [plot_df.loc[plot_df["perturbation_type"] == p, "edit_distance_normalized"].values for p in order]
    fig, ax = plt.subplots(figsize=(9, max(6, 0.4 * len(order))))
    line_color = "#333333"
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=[p.replace("_", " ") for p in order],
        patch_artist=True,
        widths=0.6,
        flierprops={"markersize": 3, "color": line_color},
        boxprops=dict(edgecolor="none"),
        medianprops=dict(color=line_color, linewidth=1.5),
        whiskerprops=dict(color=line_color),
        capprops=dict(color=line_color),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["orange"])
        patch.set_alpha(0.7)

    ax.set_xlabel("Normalized edit distance", fontsize=16)
    ax.set_ylabel("Perturbation type", fontsize=16)
    ax.set_title("Normalized edit distance by perturbation type", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf", format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 8: First divergence position by LLM (single plot, one box per model)
# ---------------------------------------------------------------------------

def plot_divergence_step_by_llm(df: pd.DataFrame, outdir: Path) -> None:
    """Single horizontal box plot: one box per LLM, y = model names, x = divergence_normalized_position."""
    _ensure_outdir(outdir)
    plot_df = df[df["divergence_normalized_position"].notna() & df["model"].notna()].copy()
    if plot_df.empty:
        return

    models = plot_df["model"].unique().tolist()
    if not models:
        return

    data = [plot_df.loc[plot_df["model"] == mod, "divergence_normalized_position"].values for mod in models]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.5 * len(models))))
    line_color = "#333333"
    bp = ax.boxplot(
        data,
        vert=False,
        tick_labels=models,
        patch_artist=True,
        widths=0.6,
        flierprops={"markersize": 3, "color": line_color},
        boxprops=dict(edgecolor="none"),
        medianprops=dict(color=line_color, linewidth=1.5),
        whiskerprops=dict(color=line_color),
        capprops=dict(color=line_color),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS["blue"])
        patch.set_alpha(0.7)
    ax.set_xlabel(DIVERGENCE_POSITION_LABEL, fontsize=16)
    ax.set_ylabel("Model (LLM)", fontsize=16)
    ax.set_title("First divergence position by LLM", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.set_xlim(-0.02, 0.8)
    fig.tight_layout()
    fig.savefig(outdir / "fig_divergence_step_by_llm.pdf", format="pdf", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_LLM_ROOTS = [
    "data/analysis_val_gpt",
    "data/analysis_val_llama",
    "data/analysis_val_qwen",
]


def _build_runs_flat_for_root(root: Path) -> pd.DataFrame:
    """Load one analysis root from runs_flat.csv and optional token_usage.csv."""
    runs_path = root / "runs_flat.csv"
    if not runs_path.exists():
        print(f"[WARN] Missing {runs_path}")
        return pd.DataFrame()

    runs_flat = pd.read_csv(runs_path)

    for col in ["reroutes", "divergence_normalized_position", "edit_distance_normalized"]:
        if col in runs_flat.columns:
            runs_flat[col] = pd.to_numeric(runs_flat[col], errors="coerce")

    for col in ["early_termination", "extended_execution"]:
        if col in runs_flat.columns:
            runs_flat[col] = _coerce_bool(runs_flat[col])

    token_path = root / "token_usage.csv"
    if token_path.exists():
        token_df = pd.read_csv(token_path)
        keep_cols = [
            "task_id",
            "perturbation_type",
            "file_type",
            "extra_tokens",
            "baseline_tokens",
            "perturbed_tokens",
        ]
        have_cols = [c for c in keep_cols if c in token_df.columns]
        if have_cols:
            runs_flat = runs_flat.merge(
                token_df[have_cols],
                on=["task_id", "perturbation_type"],
                how="left",
            )

    for col in ["extra_tokens", "baseline_tokens", "perturbed_tokens"]:
        if col in runs_flat.columns:
            runs_flat[col] = pd.to_numeric(runs_flat[col], errors="coerce")
        else:
            runs_flat[col] = np.nan

    if "file_type" not in runs_flat.columns:
        runs_flat["file_type"] = np.nan

    runs_flat["token_ratio"] = np.where(
        (runs_flat["baseline_tokens"] != 0) & runs_flat["baseline_tokens"].notna(),
        runs_flat["perturbed_tokens"] / runs_flat["baseline_tokens"],
        np.nan,
    )

    runs_flat["model"] = _infer_model_name(root)
    return runs_flat


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-quality plots from analysis CSV files.")
    parser.add_argument(
        "--roots",
        type=str,
        nargs="*",
        default=None,
        help="Analysis directories containing runs_flat.csv (and optional token_usage.csv). Combined for plots 1–4; same roots used for model comparison in plot 5.",
    )
    parser.add_argument("--outdir", type=str, default="output/plots", help="Output directory for figures")
    args = parser.parse_args()

    setup_paper_style()
    roots = args.roots
    if roots is None or roots == []:
        roots = DEFAULT_LLM_ROOTS
    roots = [Path(p).expanduser() for p in roots]

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    # Combined data from all roots (for plots 1–4: totals across models)
    print(f"[INFO] Loading {len(roots)} analysis root(s) for combined data...")
    combined_dfs: List[pd.DataFrame] = []
    for r in roots:
        if not r.exists():
            print(f"[WARN] Root not found: {r}")
            continue
        rf = _build_runs_flat_for_root(r)
        if not rf.empty:
            combined_dfs.append(rf)
    if not combined_dfs:
        print("[INFO] No valid analysis CSV data found.")
        return

    runs_flat = pd.concat(combined_dfs, ignore_index=True)
    if runs_flat.empty:
        print("[INFO] No valid analysis CSV data found.")
        return

    # Ensure extra_tokens exists (for plot 4)
    if "extra_tokens" not in runs_flat.columns:
        runs_flat["extra_tokens"] = np.nan

    # Non-LLM-comparison plots: save as combined-* (aggregated) and [llm-name]-* (per root)
    def run_non_llm_plots(data: pd.DataFrame, prefix: str) -> None:
        plot_control_flow_by_file_type(data, outdir, file_prefix=prefix)
        plot_divergence_step_by_file_type(data, outdir, file_prefix=prefix)
        plot_heatmap_file_type_manifestation(data, outdir, file_prefix=prefix)
        if data["extra_tokens"].notna().any():
            plot_extra_tokens_by_file_type(data, outdir, file_prefix=prefix)
            plot_extra_tokens_by_perturbation(data, outdir, file_prefix=prefix)
        plot_divergence_step_by_perturbation(data, outdir, file_prefix=prefix)
        if data["edit_distance_normalized"].notna().any():
            plot_edit_distance_by_perturbation(data, outdir, file_prefix=prefix)

    # Combined (aggregated) plots: [combined]-[file-name]
    print("[INFO] Generating combined (aggregated) plots: combined-*")
    run_non_llm_plots(runs_flat, "combined")

    # Per-root data (reuse for per-LLM plots and for LLM comparison)
    root_dfs: List[tuple] = []
    for r in roots:
        if not r.exists():
            continue
        rf = _build_runs_flat_for_root(r)
        if not rf.empty:
            root_dfs.append((r, rf))

    # Per-LLM plots: [llm-name]-[file-name]
    for r, rf in root_dfs:
        llm_name = str(rf["model"].iloc[0]).strip() if "model" in rf.columns and rf["model"].notna().any() else r.name
        if llm_name == r.name and r.name.startswith("[") and "]" in r.name:
            llm_name = r.name.split("]", 1)[-1].strip() or r.name
        llm_name = _sanitize_prefix(llm_name)
        print(f"[INFO] Generating per-LLM plots for: {llm_name}")
        run_non_llm_plots(rf, llm_name)

    # LLM comparison plots (keep current names, no prefix)
    print("[INFO] Plot 5: LLM × control-flow prevalence")
    llm_dfs = [rf for _, rf in root_dfs]
    if llm_dfs:
        llm_flat = pd.concat(llm_dfs, ignore_index=True)
        plot_llm_control_flow_prevalence(llm_flat, outdir)
        print("[INFO] Plot 8: first divergence position by LLM")
        plot_divergence_step_by_llm(llm_flat, outdir)
    else:
        print("[INFO] Plot 5: skipped (no data from roots)")

    print(f"[INFO] Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
