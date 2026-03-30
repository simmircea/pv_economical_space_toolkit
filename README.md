# PV Economical Space Plotting Toolkit

Small, reproducible subset of `reno` focused on PV sizing economic-space
analysis and plotting.

## What this includes

- Plot generation scripts:
  - `plot_economical_space.py`
  - `plot_cost_space.py`
- Optional data-generation scripts:
  - `pv_sizing_create_cache.py`
  - `pv_sizing_space_evaluator.py`
  - `pv_sizing_npv.py`
- Core modules under `source/` needed for loading, evaluation, and plotting.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick start
Generate plots:

```bash
python plot_economical_space.py
python plot_cost_space.py
```

If needed, regenerate cache/economical space first:

```bash
python pv_sizing_create_cache.py
python pv_sizing_space_evaluator.py