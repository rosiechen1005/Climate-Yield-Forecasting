# Climate–Yield Forecasting

Spatiotemporal machine learning pipeline forecasting detrended corn yield across the U.S. Corn Belt using multi-decade climate data.

## Method
- Built end-to-end ETL pipeline integrating 50+ years of USDA + meteorological data (1M+ records)
- Clustered regional climate regimes (k-means)
- Engineered seasonal aggregates and anomaly-based climate features
- Trained Random Forest + stacked ensemble (RF → XGB → MLP)
- Applied PCA dimensionality reduction, MICE imputation, and hyperparameter optimization

## Results
- R² improved from **0.30 → 0.76** (+150%)
- Consistent performance across 3 climate clusters
- Key drivers: seasonal Tmax & extreme heat events
- Translated model outputs into interpretable climate drivers supporting ongoing research publication

## Tools
Python • Scikit-learn • XGBoost • GeoPandas • PCA • HPC (Zaratan)

Developed during DAWN Internship, ESSIC – University of Maryland (USDA NIFA-funded)
