#!/usr/bin/env python3
"""
Aggregate divergence reports across task folders and generate summary tables + plots.

Expected layout (example):
results_root/
  <task_id_1>/
    divergence_report.json
    other_report.json
  <task_id_2>/
    report_a.json
  ...

Each JSON file may contain:
- a single divergence report dict
- a list of divergence report dicts

Outputs:
- analysis_out/runs_flat.csv
- analysis_out/by_perturbation.csv
- analysis_out/by_task.csv
- analysis_out/fig_manifestation_scatter.png
- analysis_out/fig_first_divergence_by_perturbation.png
- analysis_out/fig_control_flow_fingerprints.png
- analysis_out/fig_manifestation_counts.png
- analysis_out/fig_extra_tokens_by_perturbation.png
- analysis_out/fig_extra_tokens_by_file_type.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


# -----------------------------
# Paper-quality style configuration
# -----------------------------

def setup_paper_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')

    mpl.rcParams.update({
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
        'font.size': 9,
        'axes.titlesize': 10,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,

        # Figure settings
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Line and marker settings
        'lines.linewidth': 1.0,
        'lines.markersize': 5,

        # Axes settings
        'axes.linewidth': 0.5,
        'axes.grid': True,
        'grid.linewidth': 0.3,
        'grid.alpha': 0.3,

        # Legend
        'legend.frameon': False,
        'legend.borderpad': 0.3,

        # Ticks
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })


# Colorblind-friendly palette (IBM Design)
COLORS = {
    'blue': '#648FFF',
    'purple': '#785EF0',
    'magenta': '#DC267F',
    'orange': '#FE6100',
    'yellow': '#FFB000',
    'gray': '#808080',
    'light_gray': '#D3D3D3',
}

# File type colors
FILE_TYPE_COLORS = {
    'tabular': '#648FFF',
    'document': '#785EF0',
    'image': '#DC267F',
    'audio': '#FE6100',
    'any': '#808080',
    'unknown': '#D3D3D3',
    'data': '#FFB000',
}


# -----------------------------
# Parsing helpers
# -----------------------------

REPORT_REQUIRED_KEYS = {"task_id", "perturbation_type", "edit_distance", "control_flow_changes"}


def is_divergence_report_obj(obj: Any) -> bool:
    return isinstance(obj, dict) and REPORT_REQUIRED_KEYS.issubset(set(obj.keys()))


def safe_get(d: Dict[str, Any], *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_reports_from_json(path: Path) -> List[Dict[str, Any]]:
    """Load one JSON file and return a list of valid divergence report dicts."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[WARN] Failed to parse {path}: {e}")
        return []

    reports: List[Dict[str, Any]] = []

    if is_divergence_report_obj(data):
        reports.append(data)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if is_divergence_report_obj(item):
                reports.append(item)
            else:
                # skip non-report items silently
                pass
    else:
        # not a report file
        pass

    return reports


def flatten_report(report: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
    ed = report.get("edit_distance", {}) or {}
    cf = report.get("control_flow_changes", {}) or {}

    baseline_answer = report.get("baseline_answer")
    perturbed_answer = report.get("perturbed_answer")

    row = {
        # IDs / provenance
        "task_id": report.get("task_id"),
        "perturbation_type": report.get("perturbation_type"),
        "source_file": str(source_file),
        "source_folder": str(source_file.parent.name),

        # Primary divergence metrics
        "edit_distance": ed.get("distance"),
        "edit_distance_normalized": ed.get("normalized"),
        "substitutions": ed.get("substitutions"),
        "insertions": ed.get("insertions"),
        "deletions": ed.get("deletions"),
        "baseline_length": ed.get("baseline_length"),
        "perturbed_length": ed.get("perturbed_length"),

        # First divergence
        "first_divergence_step": report.get("first_divergence_step"),
        "first_divergence_type": report.get("first_divergence_type"),
        "divergence_normalized_position": report.get("divergence_normalized_position"),

        # Trace stats
        "baseline_steps": report.get("baseline_steps"),
        "perturbed_steps": report.get("perturbed_steps"),
        "baseline_events": report.get("baseline_events"),
        "perturbed_events": report.get("perturbed_events"),
        "baseline_routing_turns": report.get("baseline_routing_turns"),
        "perturbed_routing_turns": report.get("perturbed_routing_turns"),
        "baseline_tool_calls": report.get("baseline_tool_calls"),
        "perturbed_tool_calls": report.get("perturbed_tool_calls"),

        # Control-flow changes
        "reroutes": cf.get("reroutes"),
        "extra_routing_cycles": cf.get("extra_routing_cycles"),
        "missing_routing_cycles": cf.get("missing_routing_cycles"),
        "tool_failures_introduced": cf.get("tool_failures_introduced"),
        "tool_failures_avoided": cf.get("tool_failures_avoided"),
        "tool_calls_added": cf.get("tool_calls_added"),
        "tool_calls_removed": cf.get("tool_calls_removed"),
        "early_termination": cf.get("early_termination"),
        "extended_execution": cf.get("extended_execution"),
        "total_changes": cf.get("total_changes"),
        "tool_state_changes": cf.get("tool_state_changes"),

        # Outcome
        "answer_changed": report.get("answer_changed"),
        "baseline_answer_is_null": baseline_answer is None,
        "perturbed_answer_is_null": perturbed_answer is None,
        "baseline_answer_len": len(str(baseline_answer)) if baseline_answer is not None else 0,
        "perturbed_answer_len": len(str(perturbed_answer)) if perturbed_answer is not None else 0,
    }

    return row


# -----------------------------
# Derived labels (manifestation taxonomy)
# -----------------------------

def classify_manifestation(row: pd.Series) -> str:
    """
    Simple rule-based manifestation taxonomy.

    Priority order matters.
    """
    answer_changed = bool(row.get("answer_changed", False))
    perturbed_null = bool(row.get("perturbed_answer_is_null", False))
    early_term = bool(row.get("early_termination", False))
    extended = bool(row.get("extended_execution", False))
    reroutes = (row.get("reroutes") or 0)
    total_changes = (row.get("total_changes") or 0)
    edn = row.get("edit_distance_normalized")
    edn = float(edn) if edn is not None and not pd.isna(edn) else None

    # catastrophic / failure-like first
    if perturbed_null and (early_term or (edn is not None and edn >= 0.5)):
        return "catastrophic_failure"

    if early_term:
        return "early_termination"

    if extended and (total_changes >= 3 or (edn is not None and edn >= 0.2)):
        return "loop_or_extended_execution"

    # silent corruption: outcome changes with no structural signal
    if answer_changed and (edn == 0 or (edn is not None and edn <= 1e-9)) and total_changes == 0:
        return "silent_semantic_corruption"

    # strategy / routing shift
    if reroutes > 0:
        return "strategy_reroute"

    # structural divergence without answer flip
    if (edn is not None and edn > 0) or total_changes > 0:
        if answer_changed:
            return "structural_divergence_with_outcome_change"
        return "structural_divergence_recovered"

    # no observable effect
    if not answer_changed:
        return "no_observable_effect"

    # fallback
    return "outcome_change_uncategorized"


# -----------------------------
# Aggregations
# -----------------------------

def build_aggregates(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    # Coerce bools / numerics
    bool_cols = ["answer_changed", "early_termination", "extended_execution",
                 "baseline_answer_is_null", "perturbed_answer_is_null"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].fillna(False).astype(bool)

    numeric_cols = [
        "edit_distance", "edit_distance_normalized", "substitutions", "insertions", "deletions",
        "first_divergence_step", "divergence_normalized_position",
        "baseline_steps", "perturbed_steps", "baseline_events", "perturbed_events",
        "baseline_routing_turns", "perturbed_routing_turns",
        "baseline_tool_calls", "perturbed_tool_calls",
        "reroutes", "extra_routing_cycles", "missing_routing_cycles",
        "tool_failures_introduced", "tool_failures_avoided",
        "tool_calls_added", "tool_calls_removed",
        "total_changes", "tool_state_changes"
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived deltas
    df["delta_steps"] = df["perturbed_steps"] - df["baseline_steps"]
    df["delta_events"] = df["perturbed_events"] - df["baseline_events"]
    df["delta_tool_calls"] = df["perturbed_tool_calls"] - df["baseline_tool_calls"]
    df["delta_routing_turns"] = df["perturbed_routing_turns"] - df["baseline_routing_turns"]

    # Manifestation label
    df["manifestation_type"] = df.apply(classify_manifestation, axis=1)

    # Helpful booleans
    df["has_structural_divergence"] = df["edit_distance_normalized"].fillna(0) > 0
    df["has_reroute"] = df["reroutes"].fillna(0) > 0
    df["has_total_changes"] = df["total_changes"].fillna(0) > 0

    # Per-perturbation aggregation
    def pct(series: pd.Series) -> float:
        s = series.dropna()
        if len(s) == 0:
            return float("nan")
        return 100.0 * s.astype(bool).mean()

    by_p = (
        df.groupby("perturbation_type", dropna=False)
        .apply(lambda g: pd.Series({
            "n_runs": len(g),
            "n_tasks": g["task_id"].nunique(),
            "answer_changed_rate_pct": pct(g["answer_changed"]),
            "perturbed_null_rate_pct": pct(g["perturbed_answer_is_null"]),
            "reroute_rate_pct": pct(g["has_reroute"]),
            "early_termination_rate_pct": pct(g["early_termination"]),
            "extended_execution_rate_pct": pct(g["extended_execution"]),
            "mean_edit_distance_normalized": g["edit_distance_normalized"].mean(),
            "median_edit_distance_normalized": g["edit_distance_normalized"].median(),
            "mean_total_changes": g["total_changes"].mean(),
            "median_total_changes": g["total_changes"].median(),
            "mean_first_divergence_pos": g["divergence_normalized_position"].mean(),
            "median_first_divergence_pos": g["divergence_normalized_position"].median(),
            "mean_delta_steps": g["delta_steps"].mean(),
            "mean_delta_tool_calls": g["delta_tool_calls"].mean(),
            "silent_semantic_corruption_count": (g["manifestation_type"] == "silent_semantic_corruption").sum(),
            "strategy_reroute_count": (g["manifestation_type"] == "strategy_reroute").sum(),
            "catastrophic_failure_count": (g["manifestation_type"] == "catastrophic_failure").sum(),
        }))
        .reset_index()
        .sort_values(["answer_changed_rate_pct", "mean_edit_distance_normalized"], ascending=[False, False])
    )

    # Per-task aggregation (useful for debugging task-specific fragility)
    by_task = (
        df.groupby("task_id", dropna=False)
        .apply(lambda g: pd.Series({
            "n_runs": len(g),
            "n_perturbations": g["perturbation_type"].nunique(),
            "answer_changed_rate_pct": pct(g["answer_changed"]),
            "perturbed_null_rate_pct": pct(g["perturbed_answer_is_null"]),
            "mean_edit_distance_normalized": g["edit_distance_normalized"].mean(),
            "median_edit_distance_normalized": g["edit_distance_normalized"].median(),
            "mean_total_changes": g["total_changes"].mean(),
            "reroute_rate_pct": pct(g["has_reroute"]),
            "early_termination_rate_pct": pct(g["early_termination"]),
            "extended_execution_rate_pct": pct(g["extended_execution"]),
        }))
        .reset_index()
        .sort_values(["answer_changed_rate_pct", "mean_edit_distance_normalized"], ascending=[False, False])
    )

    return {"runs_flat": df, "by_perturbation": by_p, "by_task": by_task}


def extract_token_data(root: Path) -> pd.DataFrame:
    """
    Extract token usage data from summary.json and file type from details.json.

    Returns a DataFrame with:
    - task_id, perturbation_type, baseline_tokens, perturbed_tokens, extra_tokens, file_type
    """
    rows: List[Dict[str, Any]] = []

    # Find all task folders (containing summary.json)
    summary_files = sorted(root.rglob("summary.json"))

    for summary_path in summary_files:
        try:
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to parse {summary_path}: {e}")
            continue

        if not isinstance(summary, dict):
            continue

        task_id = summary.get("task_id")
        baseline = summary.get("baseline", {})
        baseline_tokens = baseline.get("total_tokens") if baseline else None
        perturbations = summary.get("perturbations", [])

        # Try to get file type from details.json in the same folder
        details_path = summary_path.parent / "details.json"
        file_type = None
        file_name = None
        if details_path.exists():
            try:
                with details_path.open("r", encoding="utf-8") as f:
                    details = json.load(f)
                task_meta = details.get("task", {}).get("metadata", {})
                file_inspection = task_meta.get("file_inspection", {})
                file_type = file_inspection.get("file_type")
                file_name = file_inspection.get("file_name") or task_meta.get("file_name")
            except Exception:
                pass

        # Infer file type from perturbation type if not available
        def infer_file_type_from_perturbation(ptype: str) -> Optional[str]:
            tabular_perturbations = {"column_swap", "label_corrupt", "data_type_corrupt",
                                     "row_duplicate", "irrelevant_columns", "unit_change"}
            image_perturbations = {"blur", "noise", "low_resolution", "partial_occlusion",
                                   "contrast_reduction", "watermark"}
            audio_perturbations = {"background_noise", "speed_change", "low_pass_filter"}
            document_perturbations = {"ocr_noise", "number_corruption", "text_redaction",
                                      "paragraph_shuffle", "encoding_error", "section_removal"}

            if ptype in tabular_perturbations:
                return "tabular"
            elif ptype in image_perturbations:
                return "image"
            elif ptype in audio_perturbations:
                return "audio"
            elif ptype in document_perturbations:
                return "document"
            elif ptype == "null_content":
                return "any"
            return None

        for pert in perturbations:
            if not isinstance(pert, dict):
                continue

            ptype = pert.get("type")
            perturbed_tokens = pert.get("total_tokens")

            if baseline_tokens is not None and perturbed_tokens is not None:
                extra_tokens = perturbed_tokens - baseline_tokens
            else:
                extra_tokens = None

            # Use file type from details.json, or infer from perturbation type
            effective_file_type = file_type or infer_file_type_from_perturbation(ptype)

            rows.append({
                "task_id": task_id,
                "perturbation_type": ptype,
                "baseline_tokens": baseline_tokens,
                "perturbed_tokens": perturbed_tokens,
                "extra_tokens": extra_tokens,
                "file_type": effective_file_type,
                "file_name": file_name,
                "source_file": str(summary_path),
            })

    return pd.DataFrame(rows)


def extract_baseline_answers(root: Path) -> pd.DataFrame:
    # load all files named summary.json
    json_files = sorted(root.rglob("summary.json"))

    # generate a json with key as task_id which should include
    # question, expected answer and baseline details
    dic = {}
    for jf in json_files:
        try:
            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to parse {jf}: {e}")
            continue

        if not isinstance(data, dict):
            continue

        task_id = data.get("task_id")
        question = data.get("question")
        expected_answer = data.get("expected_answer")
        baseline_answer = data.get("baseline")

        if task_id is not None:
            dic[task_id] = {
                "question": question,
                "expected_answer": expected_answer,
                "baseline": baseline_answer
            }

    return dic

# -----------------------------
# Plotting
# -----------------------------

def _ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def plot_manifestation_scatter(df: pd.DataFrame, outdir: Path):
    """
    x = normalized edit distance
    y = answer_changed (0/1)
    jittered y for readability, colored by answer_changed
    """
    _ensure_outdir(outdir)

    plot_df = df.copy()
    if plot_df.empty:
        return

    plot_df["x"] = plot_df["edit_distance_normalized"].fillna(0.0)
    plot_df["y"] = plot_df["answer_changed"].astype(int)

    # deterministic jitter
    jitters = []
    for i in range(len(plot_df)):
        jitters.append(((i % 11) - 5) * 0.025)
    plot_df["y_jitter"] = plot_df["y"] + pd.Series(jitters, index=plot_df.index)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Two-color scheme based on answer_changed
    colors = [COLORS['magenta'] if v else COLORS['blue'] for v in plot_df["answer_changed"]]

    ax.scatter(
        plot_df["x"], plot_df["y_jitter"],
        c=colors,
        alpha=0.6,
        s=25,
        edgecolors='none'
    )

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['magenta'],
               markersize=7, label='Answer changed'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['blue'],
               markersize=7, label='Answer unchanged'),
    ]
    ax.legend(handles=legend_elements, loc='upper left')

    ax.set_xlabel("Normalized edit distance")
    ax.set_ylabel("Answer changed")
    ax.set_title("Structural divergence vs outcome change")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.3, 1.3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['No', 'Yes'])

    fig.tight_layout()
    fig.savefig(outdir / "fig_manifestation_scatter.pdf", format='pdf')
    fig.savefig(outdir / "fig_manifestation_scatter.png", format='png')
    plt.close(fig)


def plot_first_divergence_by_perturbation(df: pd.DataFrame, outdir: Path, top_n: int = 15):
    _ensure_outdir(outdir)

    plot_df = df.copy()
    plot_df = plot_df[plot_df["divergence_normalized_position"].notna()]

    if plot_df.empty:
        return

    order = (
        plot_df.groupby("perturbation_type")["divergence_normalized_position"]
        .median()
        .sort_values()
        .index.tolist()
    )

    if len(order) > top_n:
        # keep most frequent perturbations among all rows
        freq = plot_df["perturbation_type"].value_counts()
        keep = set(freq.index[:top_n])
        plot_df = plot_df[plot_df["perturbation_type"].isin(keep)]
        order = [p for p in order if p in keep]

    fig, ax = plt.subplots(figsize=(7, max(4, 0.4 * len(order))))

    data = [plot_df.loc[plot_df["perturbation_type"] == p, "divergence_normalized_position"].values for p in order]
    bp = ax.boxplot(data, vert=False, tick_labels=order, patch_artist=True,
                   widths=0.6, flierprops={'markersize': 3})

    for patch in bp['boxes']:
        patch.set_facecolor(COLORS['purple'])
        patch.set_alpha(0.7)

    ax.set_xlabel("First divergence position (normalized)")
    ax.set_title("Divergence onset by perturbation type")
    ax.set_xlim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(outdir / "fig_first_divergence_by_perturbation.pdf", format='pdf')
    fig.savefig(outdir / "fig_first_divergence_by_perturbation.png", format='png')
    plt.close(fig)


def plot_control_flow_fingerprints(by_perturbation: pd.DataFrame, outdir: Path, top_n: int = 12):
    _ensure_outdir(outdir)

    if by_perturbation.empty:
        return

    plot_df = by_perturbation.copy()
    plot_df = plot_df.sort_values("n_runs", ascending=False).head(top_n)
    plot_df = plot_df.sort_values("answer_changed_rate_pct", ascending=True)

    labels = plot_df["perturbation_type"].tolist()
    y = range(len(labels))

    metrics = [
        ("reroute_rate_pct", "Reroute", COLORS['blue']),
        ("early_termination_rate_pct", "Early termination", COLORS['purple']),
        ("extended_execution_rate_pct", "Extended execution", COLORS['orange']),
        ("perturbed_null_rate_pct", "Null perturbed answer", COLORS['magenta']),
    ]

    fig, ax = plt.subplots(figsize=(7, max(4.5, 0.45 * len(labels))))

    # clustered horizontal bars
    bar_h = 0.18
    offsets = [-0.27, -0.09, 0.09, 0.27]

    for (col, name, color), off in zip(metrics, offsets):
        vals = plot_df[col].fillna(0).values
        ax.barh([yi + off for yi in y], vals, height=bar_h, label=name, color=color, alpha=0.7)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Rate (%)")
    ax.set_title("Control-flow fingerprint by perturbation type")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(outdir / "fig_control_flow_fingerprints.pdf", format='pdf')
    fig.savefig(outdir / "fig_control_flow_fingerprints.png", format='png')
    plt.close(fig)


def plot_manifestation_counts(df: pd.DataFrame, outdir: Path):
    _ensure_outdir(outdir)
    if df.empty:
        return

    counts = df["manifestation_type"].value_counts().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(6, max(4, 0.4 * len(counts))))

    # Clean up labels for display
    display_labels = [m.replace('_', ' ') for m in counts.index.tolist()]

    ax.barh(display_labels, counts.values.tolist(), color=COLORS['orange'], alpha=0.7)
    ax.set_xlabel("Number of runs")
    ax.set_title("Manifestation categories across runs")

    # annotate counts
    for i, v in enumerate(counts.values.tolist()):
        ax.text(v + max(counts) * 0.01, i, str(int(v)), va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(outdir / "fig_manifestation_counts.pdf", format='pdf')
    fig.savefig(outdir / "fig_manifestation_counts.png", format='png')
    plt.close(fig)


def plot_extra_tokens_by_perturbation(token_df: pd.DataFrame, outdir: Path, top_n: int = 15):
    """
    Box plot of extra tokens used per perturbation type.
    Only includes runs where perturbed used MORE tokens than baseline (extra_tokens > 0).
    """
    _ensure_outdir(outdir)

    plot_df = token_df.copy()
    plot_df = plot_df[plot_df["extra_tokens"].notna() & (plot_df["extra_tokens"] > 0)]

    if plot_df.empty:
        print("[INFO] No token data available for extra tokens by perturbation plot.")
        return

    # Order by median extra tokens
    order = (
        plot_df.groupby("perturbation_type")["extra_tokens"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    # Limit to top_n perturbations by frequency
    if len(order) > top_n:
        freq = plot_df["perturbation_type"].value_counts()
        keep = set(freq.index[:top_n])
        plot_df = plot_df[plot_df["perturbation_type"].isin(keep)]
        order = [p for p in order if p in keep]

    if not order:
        return

    fig, ax = plt.subplots(figsize=(7, max(4.5, 0.4 * len(order))))

    # Prepare data for boxplot
    data = [plot_df.loc[plot_df["perturbation_type"] == p, "extra_tokens"].values for p in order]
    bp = ax.boxplot(data, vert=False, tick_labels=order, patch_artist=True,
                   widths=0.6, flierprops={'markersize': 3})

    for patch in bp["boxes"]:
        patch.set_facecolor(COLORS['orange'])
        patch.set_alpha(0.7)

    ax.set_xlabel("Extra tokens")
    ax.set_title("Extra tokens by perturbation type")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))

    fig.tight_layout()
    fig.savefig(outdir / "fig_extra_tokens_by_perturbation.pdf", format='pdf')
    fig.savefig(outdir / "fig_extra_tokens_by_perturbation.png", format='png')
    plt.close(fig)


def plot_extra_tokens_by_file_type(token_df: pd.DataFrame, outdir: Path):
    """
    Box plot of extra tokens used per file type.
    Only includes runs where perturbed used MORE tokens than baseline (extra_tokens > 0).
    """
    _ensure_outdir(outdir)

    plot_df = token_df.copy()
    plot_df = plot_df[plot_df["extra_tokens"].notna() & plot_df["file_type"].notna() & (plot_df["extra_tokens"] > 0)]

    if plot_df.empty:
        print("[INFO] No token data available for extra tokens by file type plot.")
        return

    # Order by median extra tokens
    order = (
        plot_df.groupby("file_type")["extra_tokens"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    if not order:
        return

    fig, ax = plt.subplots(figsize=(6, max(3.5, 0.5 * len(order))))

    # Prepare data for boxplot
    data = [plot_df.loc[plot_df["file_type"] == ft, "extra_tokens"].values for ft in order]
    bp = ax.boxplot(data, vert=False, tick_labels=order, patch_artist=True,
                   widths=0.5, flierprops={'markersize': 3})

    # Color boxes by file type
    for patch, ft in zip(bp["boxes"], order):
        patch.set_facecolor(FILE_TYPE_COLORS.get(ft, COLORS['gray']))
        patch.set_alpha(0.7)

    ax.set_xlabel("Extra tokens")
    ax.set_title("Extra tokens by file type")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))

    # Add count annotations
    for i, ft in enumerate(order):
        count = len(plot_df[plot_df["file_type"] == ft])
        ax.annotate(f"n={count}",
                    xy=(ax.get_xlim()[1] * 0.95, i + 1),
                    fontsize=7, va="center", ha="right")

    fig.tight_layout()
    fig.savefig(outdir / "fig_extra_tokens_by_file_type.pdf", format='pdf')
    fig.savefig(outdir / "fig_extra_tokens_by_file_type.png", format='png')
    plt.close(fig)


def plot_token_summary_bar(token_df: pd.DataFrame, outdir: Path):
    """
    Bar chart showing mean extra tokens by perturbation type with error bars.
    Only includes runs where perturbed used MORE tokens than baseline (extra_tokens > 0).
    """
    _ensure_outdir(outdir)

    plot_df = token_df.copy()
    plot_df = plot_df[plot_df["extra_tokens"].notna() & (plot_df["extra_tokens"] > 0)]

    if plot_df.empty:
        return

    # Aggregate by perturbation type
    agg = (
        plot_df.groupby("perturbation_type")["extra_tokens"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg = agg.sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(4.5, 0.35 * len(agg))))

    y_pos = range(len(agg))

    ax.barh(y_pos, agg["mean"], xerr=agg["std"], color=COLORS['orange'], alpha=0.7,
            capsize=2, error_kw={'linewidth': 0.8})
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(agg["perturbation_type"].tolist())

    ax.set_xlabel("Mean extra tokens (± std)")
    ax.set_title("Mean extra tokens by perturbation type")
    ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))

    fig.tight_layout()
    fig.savefig(outdir / "fig_extra_tokens_bar.pdf", format='pdf')
    fig.savefig(outdir / "fig_extra_tokens_bar.png", format='png')
    plt.close(fig)


# -----------------------------
# Main scan routine
# -----------------------------

def scan_reports(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    json_files = sorted(root.rglob("divergence_report.json"))

    if not json_files:
        print(f"[WARN] No JSON files found under: {root}")
        return pd.DataFrame()

    for jf in json_files:
        reports = load_reports_from_json(jf)
        if not reports:
            continue
        for rep in reports:
            row = flatten_report(rep, jf)

            # sanity check task id vs folder name (non-fatal)
            folder_task = jf.parent.name
            if row.get("task_id") and folder_task and row["task_id"] != folder_task:
                # keep both; useful for debugging
                row["task_folder_mismatch"] = True
            else:
                row["task_folder_mismatch"] = False

            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def print_quick_summary(df: pd.DataFrame):
    if df.empty:
        print("[INFO] No divergence reports parsed.")
        return

    n_runs = len(df)
    n_tasks = df["task_id"].nunique(dropna=True)
    n_ptypes = df["perturbation_type"].nunique(dropna=True)

    # make sure booleans safe
    answer_changed_rate = 100.0 * df["answer_changed"].fillna(False).astype(bool).mean() if "answer_changed" in df else float("nan")

    print("\n=== Quick Summary ===")
    print(f"Runs parsed:              {n_runs}")
    print(f"Unique tasks:             {n_tasks}")
    print(f"Perturbation types:       {n_ptypes}")
    print(f"Answer-changed rate:      {answer_changed_rate:.1f}%")

    if "edit_distance_normalized" in df and df["edit_distance_normalized"].notna().any():
        print(f"Mean norm. edit distance: {df['edit_distance_normalized'].mean():.3f}")
        print(f"Median norm. edit dist.:  {df['edit_distance_normalized'].median():.3f}")

    if "manifestation_type" in df:
        print("\nManifestation counts:")
        print(df["manifestation_type"].value_counts().to_string())


def main():
    parser = argparse.ArgumentParser(description="Aggregate divergence report JSONs and generate plots.")
    parser.add_argument("root", type=str, help="Root directory containing task-id subfolders with JSON report files")
    parser.add_argument("--outdir", type=str, default="analysis_out", help="Output directory for CSVs and figures")
    args = parser.parse_args()

    # Setup paper-quality plotting style
    setup_paper_style()

    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Scanning: {root}")
    df = scan_reports(root)

    if df.empty:
        print("[INFO] No valid reports found.")
        return

    aggs = build_aggregates(df)
    runs_flat = aggs["runs_flat"]
    by_perturbation = aggs["by_perturbation"]
    by_task = aggs["by_task"]

    # Extract baseline answers for each task (for reference/debugging)
    baseline_answers = extract_baseline_answers(root)
    filename = outdir / "baseline_answers.json"
    with open(filename, 'w') as json_file:
        json.dump(baseline_answers, json_file, indent=4)


    # Save CSVs
    runs_flat.to_csv(outdir / "runs_flat.csv", index=False)
    by_perturbation.to_csv(outdir / "by_perturbation.csv", index=False)
    by_task.to_csv(outdir / "by_task.csv", index=False)

    # Save a compact "interesting cases" table
    interesting = runs_flat.copy()
    interesting = interesting.sort_values(
        by=["answer_changed", "edit_distance_normalized", "total_changes"],
        ascending=[False, False, False]
    )
    interesting_cols = [
        "task_id", "perturbation_type", "manifestation_type",
        "answer_changed", "perturbed_answer_is_null",
        "edit_distance_normalized", "total_changes",
        "first_divergence_type", "divergence_normalized_position",
        "reroutes", "early_termination", "extended_execution",
        "tool_calls_added", "tool_calls_removed",
        "source_file"
    ]
    existing_cols = [c for c in interesting_cols if c in interesting.columns]
    interesting[existing_cols].to_csv(outdir / "interesting_cases.csv", index=False)

    # Plots
    plot_manifestation_scatter(runs_flat, outdir)
    plot_first_divergence_by_perturbation(runs_flat, outdir)
    plot_control_flow_fingerprints(by_perturbation, outdir)
    plot_manifestation_counts(runs_flat, outdir)

    # Extract and plot token usage data
    print("[INFO] Extracting token usage data...")
    token_df = extract_token_data(root)

    if not token_df.empty:
        # Save token data CSV
        token_df.to_csv(outdir / "token_usage.csv", index=False)

        # Filter to only runs where perturbed used MORE tokens than baseline
        token_df_positive = token_df[token_df["extra_tokens"].notna() & (token_df["extra_tokens"] > 0)]

        # Create token aggregate by perturbation (only positive extra tokens)
        token_by_perturbation = (
            token_df_positive
            .groupby("perturbation_type")
            .agg({
                "extra_tokens": ["mean", "median", "std", "min", "max", "count"],
                "baseline_tokens": "mean",
                "perturbed_tokens": "mean",
            })
        )
        token_by_perturbation.columns = [
            "extra_tokens_mean", "extra_tokens_median", "extra_tokens_std",
            "extra_tokens_min", "extra_tokens_max", "n_runs",
            "baseline_tokens_mean", "perturbed_tokens_mean"
        ]
        token_by_perturbation = token_by_perturbation.reset_index()
        token_by_perturbation = token_by_perturbation.sort_values("extra_tokens_median", ascending=False)
        token_by_perturbation.to_csv(outdir / "token_by_perturbation.csv", index=False)

        # Create token aggregate by file type (only positive extra tokens)
        token_by_file_type = (
            token_df_positive[token_df_positive["file_type"].notna()]
            .groupby("file_type")
            .agg({
                "extra_tokens": ["mean", "median", "std", "min", "max", "count"],
                "baseline_tokens": "mean",
                "perturbed_tokens": "mean",
            })
        )
        token_by_file_type.columns = [
            "extra_tokens_mean", "extra_tokens_median", "extra_tokens_std",
            "extra_tokens_min", "extra_tokens_max", "n_runs",
            "baseline_tokens_mean", "perturbed_tokens_mean"
        ]
        token_by_file_type = token_by_file_type.reset_index()
        token_by_file_type = token_by_file_type.sort_values("extra_tokens_median", ascending=False)
        token_by_file_type.to_csv(outdir / "token_by_file_type.csv", index=False)

        # Generate token plots
        plot_extra_tokens_by_perturbation(token_df, outdir)
        plot_extra_tokens_by_file_type(token_df, outdir)
        plot_token_summary_bar(token_df, outdir)

        # Print token summary
        print("\n=== Token Usage Summary ===")
        all_valid = token_df[token_df["extra_tokens"].notna()]
        positive_only = token_df_positive
        if not positive_only.empty:
            pct_positive = 100.0 * len(positive_only) / len(all_valid) if len(all_valid) > 0 else 0
            print(f"Total runs with token data: {len(all_valid)}")
            print(f"Runs with extra tokens > 0: {len(positive_only)} ({pct_positive:.1f}%)")
            print(f"Mean extra tokens:          {positive_only['extra_tokens'].mean():,.0f}")
            print(f"Median extra tokens:        {positive_only['extra_tokens'].median():,.0f}")
            print(f"Max extra tokens:           {positive_only['extra_tokens'].max():,.0f}")
    else:
        print("[INFO] No token data found in summary.json files.")

    print_quick_summary(runs_flat)
    print(f"\n[INFO] Outputs written to: {outdir}")


if __name__ == "__main__":
    main()
