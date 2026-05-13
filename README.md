# Trace-Level Analysis of Information Contamination in Multi-Agent Systems


This repository reproduces the statistics and figures used in the paper. It contains scripts and data for generating the reported results across the evaluated models.

Paper: https://arxiv.org/abs/2604.27586

## Contents

```
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── reproduce.sh            # Master reproduction script
├── data/                   # Per-model analysis directories
│   ├── analysis_val_gpt/
│   ├── analysis_val_llama/
│   └── analysis_val_qwen/
├── scripts/
│   ├── paper_stats.py      # Generate statistics
│   └── paper_plots.py      # Plot generator from analysis CSVs
├── reference/              # Example expected outputs for verification
│   ├── paper_stats.txt
│   └── paper_stats_gpt.txt
└── output/                 # Generated outputs (after running reproduce.sh)
```

## Requirements

- Python 3.9+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

## Quick Start

```bash
# Reproduce all statistics and figures
./reproduce.sh
```

## What This Artifact Reproduces

### Statistics (from `scripts/paper_stats.py`)

| Stats JSON | Contents |
|-----------|----------|
| `paper_stats_combined.json` | Cross-model totals, manifestation breakdowns, token overhead summaries, and file-type aggregates |
| `paper_stats_gpt.json` | GPT-only totals, token overhead analysis, and top-5 high-cost perturbations |

### Plots (from `scripts/paper_plots.py`)

| Plot | Description |
|------|-------------|
| `gpt-5-mini-fig_edit_distance_by_perturbation.pdf` | Edit distance by perturbation |
| `gpt-5-mini-fig_divergence_step_by_perturbation.pdf` | Divergence step by perturbation |
| `gpt-5-mini-fig_extra_tokens_by_perturbation.pdf` | Extra tokens by perturbation |
| `gpt-5-mini-fig_control_flow_by_file_type.pdf` | Control-flow prevalence by file type |
| `gpt-5-mini-fig_divergence_step_by_file_type.pdf` | Divergence step by file type |
| `fig_divergence_step_by_llm.pdf` | Divergence position by model |
| `fig_llm_control_flow_prevalence.pdf` | Cross-model control-flow comparison |

## Manual Reproduction

### Generate Statistics

**Combined (3 models):**
```bash
python scripts/paper_stats.py \
    data/analysis_val_gpt data/analysis_val_llama data/analysis_val_qwen \
    --output output/paper_stats_combined.json
```

**GPT only (for cost analysis):**
```bash
python scripts/paper_stats.py \
    data/analysis_val_gpt \
    --output output/paper_stats_gpt.json
```

### Generate Plots

```bash
python scripts/paper_plots.py \
    --roots data/analysis_val_gpt data/analysis_val_llama data/analysis_val_qwen \
    --outdir output/plots
```

## Data Description

Each `data/analysis_val_*/` directory contains:

| File | Description |
|------|-------------|
| `runs_flat.csv` | Per-run divergence metrics (edit distance, manifestation type, control flow) |
| `token_usage.csv` | Token counts for baseline vs perturbed runs |
| `by_perturbation.csv` | Statistics aggregated by perturbation type |

### Column Definitions

**runs_flat.csv:**
- `task_id`: GAIA benchmark task identifier
- `perturbation_type`: Type of corruption applied (e.g., `column_swap`, `ocr_noise`)
- `edit_distance_normalized`: Structural divergence [0-1] using Wagner-Fischer algorithm
- `manifestation_type`: Classified outcome (e.g., `silent_semantic_corruption`, `loop_or_extended_execution`)
- `reroutes`: Number of agent routing changes
- `early_termination`: Boolean, run ended before baseline
- `extended_execution`: Boolean, run continued beyond baseline

**token_usage.csv:**
- `baseline_tokens`: Token count for clean run
- `perturbed_tokens`: Token count for perturbed run
- `file_type`: Input file category (tabular, document, image, audio)

## Notes

- Edit distance normalized to [0,1]: `distance / max(len_baseline, len_perturbed)`
- Manifestation types are mutually exclusive categories
