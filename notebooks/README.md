Notebooks for Titanic EDA
==========================

Contents
--------
- `01_eda_overview.ipynb`: Core exploratory analysis — structure, missingness,
  target distribution, key feature relationships (Sex, Pclass, Age, Fare, Embarked),
  simple engineered signals (FamilySize, IsAlone).
- `02_eda_features_and_leakage.ipynb`: Deeper feature exploration — Title from Name,
  Deck from Cabin, TicketGroupSize, interactions, and leakage considerations.

Data
----
The notebooks expect the standard Kaggle files under `data/`:
- `data/train.csv`
- `data/test.csv`

How to Run
----------
1. Create and activate a Python 3.9+ environment with pandas and seaborn:
   - `python -m venv .venv && source .venv/bin/activate`
   - `pip install -U pip pandas numpy matplotlib seaborn jupyter`
2. Launch Jupyter and open the notebooks:
   - `jupyter lab` or `jupyter notebook`
3. Run all cells top-to-bottom.

Notes
-----
- Plots use seaborn if available and fall back gracefully when not.
- No internet is required; data is local.
- These EDA notebooks do not perform any fitting on full data beyond descriptive stats.
