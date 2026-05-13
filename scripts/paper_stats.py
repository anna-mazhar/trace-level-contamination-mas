#!/usr/bin/env python3
"""
Extract statistics for research paper from analysis results.

Usage:
    python paper_stats.py analysis_val_*
    python paper_stats.py analysis_val_gpt analysis_val_llama analysis_val_qwen
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np


def load_all_data(analysis_dirs: List[Path]) -> Dict[str, pd.DataFrame]:
    """Load runs_flat.csv and token_usage.csv from all directories."""
    runs_list = []
    tokens_list = []

    for d in analysis_dirs:
        runs_path = d / "runs_flat.csv"
        tokens_path = d / "token_usage.csv"

        if runs_path.exists():
            df = pd.read_csv(runs_path)
            df['source_dir'] = d.name
            # Extract model name
            model = d.name.replace('analysis_val_', '').replace('analysis_test_', '')
            df['model'] = model
            runs_list.append(df)

        if tokens_path.exists():
            df = pd.read_csv(tokens_path)
            df['source_dir'] = d.name
            model = d.name.replace('analysis_val_', '').replace('analysis_test_', '')
            df['model'] = model
            tokens_list.append(df)

    return {
        'runs': pd.concat(runs_list, ignore_index=True) if runs_list else pd.DataFrame(),
        'tokens': pd.concat(tokens_list, ignore_index=True) if tokens_list else pd.DataFrame(),
    }


def load_correctness_data(results_dirs: List[Path]) -> pd.DataFrame:
    """Load correctness data from summary.json files in results directories."""
    rows = []

    for results_dir in results_dirs:
        for summary_path in results_dir.rglob("summary.json"):
            try:
                with open(summary_path) as f:
                    d = json.load(f)

                task_id = d.get('task_id')
                baseline = d.get('baseline', {})
                baseline_correct = baseline.get('is_correct', False) if baseline else False

                for p in d.get('perturbations', []):
                    rows.append({
                        'task_id': task_id,
                        'perturbation_type': p.get('type'),
                        'is_correct': p.get('is_correct', False),
                        'baseline_correct': baseline_correct,
                        'answer_changed': p.get('answer_changed', False),
                    })
            except Exception:
                continue

    return pd.DataFrame(rows)


def extract_model_name(source_dir: str) -> str:
    """Extract clean model name."""
    return source_dir.replace('analysis_val_', '').replace('analysis_test_', '')


def compute_paper_stats(runs_df: pd.DataFrame, tokens_df: pd.DataFrame,
                        correctness_df: pd.DataFrame = None) -> Dict[str, Any]:
    """Compute all statistics needed for the paper."""
    stats = {}

    if correctness_df is None:
        correctness_df = pd.DataFrame()

    # Ensure boolean columns
    bool_cols = ['answer_changed', 'early_termination', 'extended_execution']
    for col in bool_cols:
        if col in runs_df.columns:
            runs_df[col] = runs_df[col].fillna(False).astype(bool)

    # ===========================================
    # 1. Basic counts
    # ===========================================
    stats['total_runs'] = len(runs_df)
    stats['total_tasks'] = runs_df['task_id'].nunique()
    stats['total_perturbation_types'] = runs_df['perturbation_type'].nunique()

    # ===========================================
    # 2. Manifestation type distribution
    # ===========================================
    manifest_counts = runs_df['manifestation_type'].value_counts()
    manifest_pct = (manifest_counts / len(runs_df) * 100).round(1)
    stats['manifestation_counts'] = manifest_counts.to_dict()
    stats['manifestation_pct'] = manifest_pct.to_dict()

    # Map to paper terminology
    # Silent semantic corruption
    silent = manifest_pct.get('silent_semantic_corruption', 0)
    stats['silent_semantic_corruption_pct'] = silent

    # Behavioral detours (strategy_reroute + loop_or_extended_execution)
    behavioral = manifest_pct.get('strategy_reroute', 0) + manifest_pct.get('loop_or_extended_execution', 0)
    stats['behavioral_detours_pct'] = behavioral

    # Combined disruption (early_termination + catastrophic_failure)
    combined = manifest_pct.get('early_termination', 0) + manifest_pct.get('catastrophic_failure', 0)
    stats['combined_disruption_pct'] = combined

    # No observable effect
    no_effect = manifest_pct.get('no_observable_effect', 0)
    stats['no_observable_effect_pct'] = no_effect

    # ===========================================
    # 3. Divergent runs analysis (exclude no_observable_effect)
    # ===========================================
    divergent = runs_df[runs_df['manifestation_type'] != 'no_observable_effect']
    n_divergent = len(divergent)
    stats['divergent_runs'] = n_divergent

    if n_divergent > 0:
        # Rerouting as primary signature
        reroute_count = (divergent['reroutes'].fillna(0) > 0).sum()
        stats['divergent_rerouting_pct'] = round(reroute_count / n_divergent * 100, 1)

        # Extended execution
        extended_count = divergent['extended_execution'].sum()
        stats['divergent_extended_pct'] = round(extended_count / n_divergent * 100, 1)

        # Early termination
        early_count = divergent['early_termination'].sum()
        stats['divergent_early_term_pct'] = round(early_count / n_divergent * 100, 1)

    # ===========================================
    # 4. Behavioral detours vs combined disruption breakdown
    # ===========================================
    # Behavioral detours = strategy_reroute + loop_or_extended_execution
    behavioral_df = runs_df[runs_df['manifestation_type'].isin(['strategy_reroute', 'loop_or_extended_execution'])]
    if len(behavioral_df) > 0:
        behavioral_reroute_pct = round((behavioral_df['reroutes'].fillna(0) > 0).sum() / len(behavioral_df) * 100, 1)
        stats['behavioral_reroute_pct'] = behavioral_reroute_pct

    # Combined disruption = early_termination + catastrophic_failure
    combined_df = runs_df[runs_df['manifestation_type'].isin(['early_termination', 'catastrophic_failure'])]
    if len(combined_df) > 0:
        combined_early_term_pct = round(combined_df['early_termination'].sum() / len(combined_df) * 100, 1)
        stats['combined_early_term_pct'] = combined_early_term_pct

    # ===========================================
    # 5. Token overhead analysis
    # ===========================================
    if not tokens_df.empty and 'extra_tokens' in tokens_df.columns:
        # Merge tokens with runs to get manifestation info
        tokens_with_manifest = tokens_df.merge(
            runs_df[['task_id', 'perturbation_type', 'manifestation_type', 'source_dir']],
            on=['task_id', 'perturbation_type', 'source_dir'],
            how='left'
        )

        # Compute token multiplier (perturbed / baseline)
        valid_tokens = tokens_with_manifest[
            (tokens_with_manifest['baseline_tokens'] > 0) &
            (tokens_with_manifest['perturbed_tokens'].notna())
        ].copy()

        if len(valid_tokens) > 0:
            valid_tokens['token_multiplier'] = valid_tokens['perturbed_tokens'] / valid_tokens['baseline_tokens']

            # Overall stats
            stats['median_token_multiplier'] = round(valid_tokens['token_multiplier'].median(), 2)
            stats['mean_token_multiplier'] = round(valid_tokens['token_multiplier'].mean(), 2)

            # By manifestation type
            token_by_manifest = {}
            for mtype in valid_tokens['manifestation_type'].dropna().unique():
                subset = valid_tokens[valid_tokens['manifestation_type'] == mtype]['token_multiplier']
                if len(subset) > 0:
                    token_by_manifest[mtype] = {
                        'median': round(subset.median(), 2),
                        'mean': round(subset.mean(), 2),
                        'q25': round(subset.quantile(0.25), 2),
                        'q75': round(subset.quantile(0.75), 2),
                        'count': len(subset),
                    }
            stats['token_by_manifestation'] = token_by_manifest

            # Extended execution overhead
            extended_tokens = valid_tokens[valid_tokens['manifestation_type'] == 'loop_or_extended_execution']
            if len(extended_tokens) > 0:
                stats['extended_median_multiplier'] = round(extended_tokens['token_multiplier'].median(), 2)
                stats['extended_iqr'] = (
                    round(extended_tokens['token_multiplier'].quantile(0.25), 2),
                    round(extended_tokens['token_multiplier'].quantile(0.75), 2)
                )

            # Silent corruption overhead
            silent_tokens = valid_tokens[valid_tokens['manifestation_type'] == 'silent_semantic_corruption']
            if len(silent_tokens) > 0:
                stats['silent_median_multiplier'] = round(silent_tokens['token_multiplier'].median(), 2)

            # Early termination overhead
            early_tokens = valid_tokens[valid_tokens['manifestation_type'] == 'early_termination']
            if len(early_tokens) > 0:
                stats['early_term_median_multiplier'] = round(early_tokens['token_multiplier'].median(), 2)

            # Behavioral detours (strategy_reroute + loop_or_extended_execution)
            behavioral_tokens = valid_tokens[valid_tokens['manifestation_type'].isin(['strategy_reroute', 'loop_or_extended_execution'])]
            if len(behavioral_tokens) > 0:
                stats['behavioral_median_multiplier'] = round(behavioral_tokens['token_multiplier'].median(), 2)
                stats['behavioral_iqr'] = (
                    round(behavioral_tokens['token_multiplier'].quantile(0.25), 2),
                    round(behavioral_tokens['token_multiplier'].quantile(0.75), 2)
                )

            # High-cost vs low-cost correctness analysis
            if not correctness_df.empty:
                # Merge token data with correctness
                tokens_with_correct = valid_tokens.merge(
                    correctness_df[['task_id', 'perturbation_type', 'is_correct']],
                    on=['task_id', 'perturbation_type'],
                    how='left'
                )

                high_cost = tokens_with_correct[tokens_with_correct['token_multiplier'] > 2.0]
                low_cost = tokens_with_correct[tokens_with_correct['token_multiplier'] < 1.2]

                if len(high_cost) > 0:
                    high_cost_correct_pct = round(high_cost['is_correct'].fillna(False).mean() * 100, 1)
                    stats['high_cost_correct_pct'] = high_cost_correct_pct
                    stats['high_cost_count'] = len(high_cost)

                if len(low_cost) > 0:
                    low_cost_incorrect_pct = round((~low_cost['is_correct'].fillna(True)).mean() * 100, 1)
                    stats['low_cost_incorrect_pct'] = low_cost_incorrect_pct
                    stats['low_cost_count'] = len(low_cost)

            # ===========================================
            # 5b. Top-5 high-cost perturbations table
            # ===========================================
            # Compute median token multiplier by perturbation
            token_by_pert = valid_tokens.groupby('perturbation_type').agg({
                'token_multiplier': 'median',
                'task_id': 'count'  # n_runs
            }).rename(columns={'task_id': 'n_runs', 'token_multiplier': 'median_overhead'})

            # Merge with correctness to get recovery rate
            if not correctness_df.empty:
                recovery_by_pert = correctness_df.groupby('perturbation_type').agg({
                    'is_correct': lambda x: round(x.fillna(False).mean() * 100, 1)
                }).rename(columns={'is_correct': 'recovery_rate'})
                token_by_pert = token_by_pert.join(recovery_by_pert, how='left')
            else:
                token_by_pert['recovery_rate'] = None

            # Sort by median overhead descending and take top 5
            token_by_pert = token_by_pert.sort_values('median_overhead', ascending=False)
            top5 = token_by_pert.head(5).reset_index()

            stats['top5_high_cost_perturbations'] = []
            for _, row in top5.iterrows():
                stats['top5_high_cost_perturbations'].append({
                    'perturbation': row['perturbation_type'],
                    'median_overhead': round(row['median_overhead'], 2),
                    'recovery_rate': row['recovery_rate'] if pd.notna(row['recovery_rate']) else 'N/A',
                    'n_runs': int(row['n_runs']),
                })

    # ===========================================
    # 6. Perturbation type analysis
    # ===========================================
    # Tabular perturbations
    tabular_perts = ['column_swap', 'label_corrupt', 'data_type_corrupt', 'row_duplicate', 'irrelevant_columns', 'unit_change']
    tabular_df = runs_df[runs_df['perturbation_type'].isin(tabular_perts)]

    if len(tabular_df) > 0:
        tabular_manifest = tabular_df['manifestation_type'].value_counts()
        tabular_manifest_pct = (tabular_manifest / len(tabular_df) * 100).round(1)
        stats['tabular_manifestation_pct'] = tabular_manifest_pct.to_dict()
        stats['tabular_top_manifestation'] = tabular_manifest.index[0] if len(tabular_manifest) > 0 else None
        stats['tabular_top_manifestation_pct'] = tabular_manifest_pct.iloc[0] if len(tabular_manifest_pct) > 0 else 0

        # Silent corruption for tabular
        stats['tabular_silent_pct'] = tabular_manifest_pct.get('silent_semantic_corruption', 0)

    # ===========================================
    # 7. By file type analysis
    # ===========================================
    if not tokens_df.empty and 'file_type' in tokens_df.columns:
        # Merge to get manifestation
        file_type_df = tokens_df.merge(
            runs_df[['task_id', 'perturbation_type', 'manifestation_type', 'source_dir']],
            on=['task_id', 'perturbation_type', 'source_dir'],
            how='left'
        )

        stats['manifestation_by_file_type'] = {}
        for ft in file_type_df['file_type'].dropna().unique():
            ft_subset = file_type_df[file_type_df['file_type'] == ft]
            ft_manifest = ft_subset['manifestation_type'].value_counts()
            ft_manifest_pct = (ft_manifest / len(ft_subset) * 100).round(1)
            stats['manifestation_by_file_type'][ft] = ft_manifest_pct.to_dict()

    # ===========================================
    # 8. By model analysis
    # ===========================================
    stats['by_model'] = {}
    for model in runs_df['model'].unique():
        model_df = runs_df[runs_df['model'] == model]
        model_manifest = model_df['manifestation_type'].value_counts()
        model_manifest_pct = (model_manifest / len(model_df) * 100).round(1)

        # Behavioral detours for this model
        behavioral_pct = model_manifest_pct.get('strategy_reroute', 0) + model_manifest_pct.get('loop_or_extended_execution', 0)

        stats['by_model'][model] = {
            'n_runs': len(model_df),
            'n_tasks': model_df['task_id'].nunique(),
            'manifestation_pct': model_manifest_pct.to_dict(),
            'behavioral_detours_pct': round(behavioral_pct, 1),
            'silent_pct': model_manifest_pct.get('silent_semantic_corruption', 0),
            'combined_disruption_pct': round(
                model_manifest_pct.get('early_termination', 0) +
                model_manifest_pct.get('catastrophic_failure', 0), 1
            ),
        }

    return stats


def print_paper_stats(stats: Dict[str, Any]):
    """Print statistics formatted for paper."""
    print("=" * 70)
    print("STATISTICS FOR RESEARCH PAPER")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("1. BASIC COUNTS")
    print("=" * 70)
    print(f"We analyze {stats['total_runs']} paired clean/perturbed runs across {stats['total_tasks']} GAIA tasks.")

    print("\n" + "=" * 70)
    print("2. MANIFESTATION DISTRIBUTION (Overall)")
    print("=" * 70)
    print(f"Silent semantic corruption: {stats['silent_semantic_corruption_pct']:.1f}%")
    print(f"Behavioral detours: {stats['behavioral_detours_pct']:.1f}%")
    print(f"Combined disruption: {stats['combined_disruption_pct']:.1f}%")
    print(f"No observable effect: {stats['no_observable_effect_pct']:.1f}%")
    print("\nDetailed breakdown:")
    for mtype, pct in sorted(stats['manifestation_pct'].items(), key=lambda x: -x[1]):
        print(f"  {mtype}: {pct:.1f}%")

    print("\n" + "=" * 70)
    print("3. DIVERGENT RUNS ANALYSIS")
    print("=" * 70)
    print(f"Total divergent runs: {stats['divergent_runs']}")
    print(f"Rerouting as primary signature: {stats.get('divergent_rerouting_pct', 'N/A')}%")
    print(f"Extended execution: {stats.get('divergent_extended_pct', 'N/A')}%")
    print(f"Early termination: {stats.get('divergent_early_term_pct', 'N/A')}%")

    print("\n" + "=" * 70)
    print("4. BEHAVIORAL DETOURS vs COMBINED DISRUPTION")
    print("=" * 70)
    print(f"Behavioral detours favor rerouting: {stats.get('behavioral_reroute_pct', 'N/A')}%")
    print(f"Combined disruptions favor early termination: {stats.get('combined_early_term_pct', 'N/A')}%")

    print("\n" + "=" * 70)
    print("5. TOKEN OVERHEAD ANALYSIS")
    print("=" * 70)
    print(f"Overall median token multiplier: {stats.get('median_token_multiplier', 'N/A')}×")

    if 'token_by_manifestation' in stats:
        print("\nBy manifestation type:")
        for mtype, tdata in sorted(stats['token_by_manifestation'].items(), key=lambda x: -x[1]['median']):
            print(f"  {mtype}:")
            print(f"    Median: {tdata['median']}×, IQR: {tdata['q25']}–{tdata['q75']}×, n={tdata['count']}")

    if 'behavioral_median_multiplier' in stats:
        print(f"\nBehavioral detours: median {stats['behavioral_median_multiplier']}× "
              f"(IQR: {stats['behavioral_iqr'][0]}–{stats['behavioral_iqr'][1]}×)")

    if 'silent_median_multiplier' in stats:
        print(f"Silent corruption: median {stats['silent_median_multiplier']}×")

    if 'early_term_median_multiplier' in stats:
        print(f"Early termination: median {stats['early_term_median_multiplier']}×")

    if 'extended_median_multiplier' in stats:
        print(f"Extended execution: median {stats['extended_median_multiplier']}× "
              f"(IQR: {stats['extended_iqr'][0]}–{stats['extended_iqr'][1]}×)")

    print("\n" + "=" * 70)
    print("5b. TOP-5 HIGH-COST PERTURBATIONS (Table~\\ref{tab:high_cost_perts})")
    print("=" * 70)
    if 'top5_high_cost_perturbations' in stats:
        print(f"{'Perturbation':<25} {'Median Overhead':>15} {'Recovery Rate':>15}")
        print("-" * 55)
        for row in stats['top5_high_cost_perturbations']:
            recovery = f"{row['recovery_rate']}%" if row['recovery_rate'] != 'N/A' else 'N/A'
            print(f"{row['perturbation']:<25} {row['median_overhead']:>14.2f}× {recovery:>15}")

    print("\n" + "=" * 70)
    print("6. TABULAR PERTURBATIONS")
    print("=" * 70)
    if 'tabular_manifestation_pct' in stats:
        print(f"Top manifestation: {stats['tabular_top_manifestation']} ({stats['tabular_top_manifestation_pct']:.1f}%)")
        print(f"Silent semantic corruption: {stats['tabular_silent_pct']:.1f}%")
        print("\nFull breakdown:")
        for mtype, pct in sorted(stats['tabular_manifestation_pct'].items(), key=lambda x: -x[1]):
            print(f"  {mtype}: {pct:.1f}%")

    print("\n" + "=" * 70)
    print("7. MANIFESTATION BY FILE TYPE")
    print("=" * 70)
    if 'manifestation_by_file_type' in stats:
        for ft, mpct in stats['manifestation_by_file_type'].items():
            print(f"\n{ft.upper()}:")
            for mtype, pct in sorted(mpct.items(), key=lambda x: -x[1]):
                print(f"  {mtype}: {pct:.1f}%")

    print("\n" + "=" * 70)
    print("8. BY MODEL ANALYSIS")
    print("=" * 70)
    for model, mdata in stats['by_model'].items():
        print(f"\n{model.upper()}:")
        print(f"  Runs: {mdata['n_runs']}, Tasks: {mdata['n_tasks']}")
        print(f"  Behavioral detours: {mdata['behavioral_detours_pct']:.1f}%")
        print(f"  Silent corruption: {mdata['silent_pct']:.1f}%")
        print(f"  Combined disruption: {mdata['combined_disruption_pct']:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Extract statistics for research paper.')
    parser.add_argument('analysis_dirs', nargs='+', help='Analysis directories')
    parser.add_argument('--results-dir', type=str, default='./results',
                        help='Results directory containing summary.json files for correctness data')
    parser.add_argument('--outdir', type=str, default='output',
                        help='Output directory for produced artifacts (default: output)')
    parser.add_argument('--output', type=str, help='Output JSON file for stats (overrides --outdir)')
    args = parser.parse_args()

    analysis_dirs = [Path(d).resolve() for d in args.analysis_dirs]

    print(f"Loading data from {len(analysis_dirs)} directories...")
    for d in analysis_dirs:
        print(f"  - {d.name}")

    data = load_all_data(analysis_dirs)

    if data['runs'].empty:
        print("[ERROR] No runs data found")
        return

    print(f"\nLoaded {len(data['runs'])} runs, {len(data['tokens'])} token records")

    # Load correctness data from results directory
    results_dir = Path(args.results_dir).resolve()
    if results_dir.exists():
        # Find all subdirectories that might contain results
        result_subdirs = [d for d in results_dir.iterdir() if d.is_dir()]
        correctness_df = load_correctness_data(result_subdirs)
        print(f"Loaded {len(correctness_df)} correctness records")
    else:
        correctness_df = pd.DataFrame()

    stats = compute_paper_stats(data['runs'], data['tokens'], correctness_df)

    print_paper_stats(stats)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.output:
        # Convert numpy types to Python types for JSON serialization
        def convert_types(obj):
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(v) for v in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w') as f:
            json.dump(convert_types(stats), f, indent=2)
        print(f"\nStats saved to: {out_path}")
    else:
        # Default output filename depends on number of analysis dirs
        if len(analysis_dirs) > 1:
            fname = 'paper_stats_combined.json'
        else:
            model = extract_model_name(analysis_dirs[0].name)
            fname = f'paper_stats_{model}.json'

        out_path = outdir / fname
        with open(out_path, 'w') as f:
            json.dump(convert_types(stats), f, indent=2)
        print(f"\nStats saved to: {out_path}")


if __name__ == '__main__':
    main()
