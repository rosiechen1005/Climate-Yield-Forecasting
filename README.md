# Climate-Yield Forecasting

## What I did
- Built end-to-end ETL pipeline integrating 50+ years of USDA + meteorological data (1M+ records)
- Clustered regional climate regimes (k-means)
- Engineered seasonal and anomaly-based climate features
- Trained Random Forest + stacked ensemble (RF → XGB → MLP)
- Applied PCA dimensionality reduction, MICE imputation, and hyperparameter optimization

## Results
- R^2 improved from **0.30 → 0.76** (+150%)
- Consistent performance across 3 climate clusters
- Key drivers: seasonal Tmax & extreme heat events

## Tools
Python • Scikit-learn • XGBoost • GeoPandas • PCA • HPC (Zaratan)

Developed during DAWN Internship, ESSIC - University of Maryland (USDA NIFA-funded)
