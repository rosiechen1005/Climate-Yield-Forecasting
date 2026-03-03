# AgriCast-ML — End-to-End Climate Forecasting System for Corn Yield

Developed during the DAWN Internship (ESSIC, University of Maryland; USDA NIFA-funded)

## Overview

Built a multi-decade climate–yield forecasting framework integrating 50+ years of USDA yield and meteorological data (1M+ records) to model Corn Belt production under heterogeneous climate regimes.

The project combined large-scale ETL, spatial joins, seasonal climate feature engineering, and cluster-specific ensemble modeling to improve predictive performance and extract interpretable climate drivers.

## Technical Contributions

- Designed an end-to-end ETL pipeline integrating USDA county-level yield data with NOAA meteorological records
- Performed station–county spatial assignment using geographic joins
- Detrended yield to isolate climate-driven variability
- Clustered regional climate regimes (k-means, 3 clusters)
- Engineered seasonal aggregates and anomaly-based extreme-heat features
- Built residual-stacked ensemble models (RF → XGBoost → MLP)
- Applied PCA dimensionality reduction and MICE imputation
- Conducted hyperparameter optimization and seasonal boundary tuning
- Executed experiments on HPC infrastructure (Zaratan cluster)

## Modeling Architecture

Baseline → Random Forest → Residual Stacking

1. Random Forest captures nonlinear climate effects  
2. XGBoost models residual structure  
3. MLP refines remaining error  

Cluster-specific configurations were applied to account for heterogeneous climate–yield relationships across regions.

## Results

- R² improved from **0.30 → 0.76** (+150%)
- Stable performance across three climate clusters
- Identified seasonal Tmax and extreme heat anomalies as dominant yield drivers
- Outputs translated into interpretable climate-risk indicators supporting ongoing research publication

## Tools

Python • Pandas • NumPy • Scikit-learn • XGBoost • GeoPandas • PCA • MICE • HPC (Zaratan)

## Data Access Note

Raw USDA and meteorological inputs are not included due to internship data agreements.  
This repository documents the modeling framework and pipeline architecture used during the DAWN Internship.
