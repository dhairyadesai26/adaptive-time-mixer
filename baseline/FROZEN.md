# ⚠️ FROZEN BASELINE — DO NOT MODIFY ⚠️

This directory contains the **exact** TimeMixer implementation from the official repository.

- **Upstream repo**: https://github.com/kwuking/TimeMixer
- **Commit SHA**: `e24610583b36fdd8c76cc17a8df4e65759a5f460`
- **Date frozen**: 2026-09-22

## Purpose

This serves as the **unmodified reference** for baseline reproduction.
All experimental comparisons use this code as the control.

## Rules

1. **Never modify** any file in this directory
2. All improvements go in `../extension/`
3. Baseline checkpoints are loaded using these exact model definitions

## Files

```
baseline/
├── layers/
│   ├── __init__.py
│   ├── Autoformer_EncDec.py   # series_decomp, moving_avg
│   ├── Embed.py               # DataEmbedding_wo_pos
│   └── StandardNorm.py        # Normalize (RevIN-style)
├── models/
│   ├── __init__.py
│   └── TimeMixer.py           # Full TimeMixer: PDM + FMM
└── FROZEN.md                  # This file
```
