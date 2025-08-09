# Kaggle Tabular Lab (ktl)

Config-first, leak-safe, reproducible pipeline for Kaggle tabular competitions.

Key ideas:
- Split first, fit transforms per-fold only.
- OOF predictions and per-fold metrics.
- Reproducible seeds, artifacted runs under `./artifacts/<timestamp>/`.

Quick start:
- Create and activate a virtualenv, then:
  - `pip install -e .`
  - `ktl --help`

Repo layout follows the provided architecture contracts. See `config/` for examples.
