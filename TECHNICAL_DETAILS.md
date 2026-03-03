# Technical Details — DAWN Corn Yield Forecasting Pipeline

This document describes repository structure, pipeline order, and reproducibility steps.

## Quick Start

1. **Raw Data**
- Place raw inputs under `data/raw/`:
   - Yield CSVs: `data/raw/corn/IA.csv`, `data/raw/corn/IL.csv`, ...
   - Meteorological CSVs: `data/raw/MET/met_IA.csv`, ...
   - `data/raw/ghcnd-stations.txt`
   - Census county shapefile under `data/raw/cb_2021_us_county_500k/`

   Alternatively, modify `RAW_DATA_DIR` in `dawn_config.py`.

2. **Data Creation**
- Run: dawn_data_creation.ipynb   
- Outputs:
  - `data/merged.csv`
  - Optional cluster CSVs (`cluster0_MI_MN_WI.csv`, etc.)

3. **Exploratory Mapping (Optional)**
- Run: dawn_mapping.ipynb
- Generates:
  - Station-level spring means
  - County maps
  - Climate cluster visualizations

4. **Model Training**
- Run: dawn_models_clean.ipynb
- Pipeline:
  - Feature engineering
  - Baseline RF
  - Engineered RF
  - Final residual-stacked model
  - Evaluation table

5. **Optional Tuning**
- Run: dawn_tuning.py
- Includes:
  - Binary/random search for seasonal date boundaries
  - Hyperparameter tuning for anomaly thresholds
  - Clear-day threshold optimization

---

## Repository Structure

| File | Purpose |
|------|----------|
| `dawn_config.py` | Shared paths and constants (`DATA_DIR`, `RAW_DATA_DIR`, yield/MET paths, `CORN_BELT_STATES`) |
| `dawn_data_creation.ipynb` | Builds `data/merged.csv`: load yield + MET, detrend, spatial join, merge |
| `dawn_mapping.ipynb` | Exploratory mapping and cluster visualization |
| `dawn_models_clean.ipynb` | Main modeling pipeline and evaluation |
| `dawn_features.py` | Feature engineering (`build_features`, seasonal aggregates, anomaly features) |
| `dawn_models.py` | Model pipelines and cluster configs (`run_baseline_rf`, `run_rf_engineered`, `run_final_model`) |
| `dawn_tuning.py` | Hyperparameter and seasonal tuning utilities |
| `data/` | Processed outputs (`merged.csv`, cluster CSVs) |

## Pipeline Order

1. Raw data ingestion  
2. Yield detrending  
3. Station–county spatial mapping  
4. Climate feature engineering  
5. Climate regime clustering  
6. Cluster-specific modeling  
7. Residual stacking  
8. Final evaluation  

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- geopandas
- shapely
- geopy
- xgboost (optional for stacking variants)

## Notes

- Designed for reproducibility and modular experimentation.
- Cluster-specific configurations stored in `CLUSTER_CONFIGS`.
- HPC execution recommended for large hyperparameter searches.
