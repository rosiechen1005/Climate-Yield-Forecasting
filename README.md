# Dawn internship — Corn yield & climate

Portfolio project: corn yield prediction from weather and seasonal features, with clustering and residual-stacking models.

## Quick start

1. **Data:** Put raw inputs under `data/raw/` (see Pipeline order below), then run `dawn_data_creation.ipynb` to create `data/merged.csv` and cluster CSVs. If you already have these files, place them in `data/`.
2. **Models:** Run `dawn_models_clean.ipynb` (it reads from `data/`).
3. **Before pushing:** Run `python clean_notebooks.py` to strip notebook outputs.

## Repo structure

| Item | Purpose |
|------|--------|
| **dawn_config.py** | Shared paths and constants: `DATA_DIR`, `RAW_DATA_DIR`, yield/MET/shapefile paths, `CORN_BELT_STATES`. |
| **dawn_data_creation.ipynb** | Builds `data/merged.csv` (and optionally cluster CSVs): load yield + MET from `data/raw/`, detrend, assign stations to counties, merge. |
| **dawn_mapping.ipynb** | Exploratory mapping: spring means by station, state/corn-belt maps, cluster map. Reads from `data/merged.csv`. |
| **dawn_models_clean.ipynb** | Main modeling: load from `data/`, feature engineering, baseline → RF → final (residual stacking), final evaluation table. |
| **dawn_features.py** | Feature engineering (`build_features`, seasonal/anomaly/clear-day). Used by models and tuning. |
| **dawn_models.py** | Model pipelines and cluster configs: `run_baseline_rf`, `run_rf_engineered`, `run_final_model`, `CLUSTER_CONFIGS`. |
| **dawn_tuning.py** | Optional tuning: binary/random search for season dates, hyperparameter search for `clear_thresh` and `anomaly_percentile`. |
| **data/** | Outputs: `merged.csv`, `cluster0_MI_MN_WI.csv`, etc. Raw inputs go under `data/raw/` (see below). |

## Pipeline order

1. **Raw data** — Place yield CSVs (`data/raw/corn/IA.csv`, ...), MET CSVs (`data/raw/MET/met_IA.csv`, ...), `data/raw/ghcnd-stations.txt`, and the Census county shapefile under `data/raw/cb_2021_us_county_500k/`. Or set `RAW_DATA_DIR` in `dawn_config.py` to another path.
2. **Data** — Run `dawn_data_creation.ipynb` to produce `data/merged.csv` and (optional) cluster CSVs in `data/`.
3. **Explore** — Optionally run `dawn_mapping.ipynb` for station/county maps.
4. **Models** — Run `dawn_models_clean.ipynb`; it reads from `data/`. Use Section 2 (tuning) to re-optimize season dates or hyperparameters if desired.

## What to put on GitHub

- **Keep:** `dawn_config.py`, `dawn_data_creation.ipynb`, `dawn_mapping.ipynb`, `dawn_models_clean.ipynb`, `dawn_features.py`, `dawn_models.py`, `dawn_tuning.py`, `clean_notebooks.py`, `.gitignore`, this README.
- **Don’t commit:** Large `data/*.csv` outputs, `data/raw/` inputs, or `dawn_models.ipynb` (original long notebook) unless you want it.

## Requirements

- Python 3.8+
- pandas, numpy, scikit-learn, geopandas, shapely, geopy (for data creation and mapping); xgboost optional for tuning variants.

## Before you push

Run `python clean_notebooks.py` to clear notebook outputs and execution counts.
