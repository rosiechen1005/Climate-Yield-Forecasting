"""
Model pipelines for corn yield prediction.
Used by dawn_models_clean.ipynb.
"""
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from dawn_features import build_features


# Cluster-specific configs (tuned); seasons: (start_month, start_day, end_month, end_day)
CLUSTER_CONFIGS = {
    0: {
        "clear_thresh": 3.18,
        "anomaly_percentile": {"prcp": 82, "tmax": 96, "tmin": 99, "snow": 99},
        "spring": (3, 13, 3, 28),
        "summer": (7, 6, 8, 31),
        "fall": (11, 4, 11, 29),
    },
    1: {
        "clear_thresh": 3.18,
        "anomaly_percentile": {"prcp": 96, "tmax": 85, "tmin": 98, "snow": 91},
        "spring": (3, 13, 3, 28),
        "summer": (7, 6, 8, 31),
        "fall": (11, 4, 11, 29),
    },
    2: {
        "clear_thresh": 2.69,
        "anomaly_percentile": {"prcp": 80, "tmax": 98, "tmin": 94, "snow": 93},
        "spring": (3, 13, 3, 28),
        "summer": (7, 6, 8, 31),
        "fall": (11, 4, 11, 29),
    },
}


def run_baseline_rf(df):
    """Baseline: raw tmax/tmin/prcp/snow means per county-year, no feature engineering."""
    feats = (
        df.groupby(["county_name", "year"])[["tmax", "tmin", "prcp", "snow"]]
        .mean()
        .reset_index()
    )
    target = (
        df[["county_name", "year", "detrended_yield"]]
        .drop_duplicates(subset=["county_name", "year"])
    )
    data = pd.merge(feats, target, on=["county_name", "year"], how="inner").dropna()
    X = data[["tmax", "tmin", "prcp", "snow"]]
    y = data["detrended_yield"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)


def run_rf_engineered(df, config):
    """RF on engineered seasonal + anomaly + clear-day features."""
    X, y = build_features(df, config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return r2_score(y_test, y_pred), mean_squared_error(y_test, y_pred)


def run_final_model(df, config):
    """Final model: residual stacking (RF + MLP on residuals). Best performer in evaluation."""
    X, y = build_features(df, config)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred_train = rf.predict(X_train)
    rf_pred_test = rf.predict(X_test)
    residual_train = y_train.values - rf_pred_train

    mlp = make_pipeline(
        StandardScaler(),
        MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=1000,
            early_stopping=True,
            learning_rate_init=0.01,
            random_state=42,
        ),
    )
    mlp.fit(X_train, residual_train)
    mlp_pred_test = mlp.predict(X_test)
    final_pred = rf_pred_test + mlp_pred_test

    return r2_score(y_test, final_pred), mean_squared_error(y_test, final_pred)
