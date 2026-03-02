"""
Feature engineering for corn yield models.
Used by dawn_models_clean.ipynb — see notebook for feature descriptions.
"""
import pandas as pd


def _season_mask(df, start_month, start_day, end_month, end_day):
    if start_month <= end_month:
        mask = (
            ((df["month"] > start_month) | ((df["month"] == start_month) & (df["day"] >= start_day)))
            & ((df["month"] < end_month) | ((df["month"] == end_month) & (df["day"] <= end_day)))
        )
    else:
        mask = (
            ((df["month"] > start_month) | ((df["month"] == start_month) & (df["day"] >= start_day)))
            | ((df["month"] < end_month) | ((df["month"] == end_month) & (df["day"] <= end_day)))
        )
    return mask


def compute_seasonal_features(df, season_name, start_month, start_day, end_month, end_day):
    mask = _season_mask(df, start_month, start_day, end_month, end_day)
    season = df[mask]
    return (
        season.groupby(["county_name", "year"])
        .agg({"tmax": "mean", "tmin": "mean", "prcp": "sum", "snow": "sum"})
        .reset_index()
        .rename(
            columns={
                "tmax": f"{season_name}_tmax_mean",
                "tmin": f"{season_name}_tmin_mean",
                "prcp": f"{season_name}_prcp_sum",
                "snow": f"{season_name}_snow_sum",
            }
        )
    )


def compute_precipitation_features(
    df, season_name, start_month, start_day, end_month, end_day,
    clear_thresh=1.0, anomaly_percentile=None,
):
    if anomaly_percentile is None:
        anomaly_percentile = {"prcp": 90, "tmax": 90, "tmin": 90, "snow": 90}
    mask = _season_mask(df, start_month, start_day, end_month, end_day)
    season = df[mask]

    prcp_thresh = season["prcp"].quantile(anomaly_percentile.get("prcp", 90) / 100)
    tmax_thresh = season["tmax"].quantile(anomaly_percentile.get("tmax", 90) / 100)
    tmin_thresh = season["tmin"].quantile(anomaly_percentile.get("tmin", 90) / 100)
    snow_thresh = season["snow"].quantile(anomaly_percentile.get("snow", 90) / 100)

    season = season.assign(
        clear_day=(season["prcp"] <= clear_thresh).astype(int),
        anomaly_prcp=(season["prcp"] >= prcp_thresh).astype(int),
        anomaly_tmax=(season["tmax"] >= tmax_thresh).astype(int),
        anomaly_tmin=(season["tmin"] >= tmin_thresh).astype(int),
        anomaly_snow=(season["snow"] >= snow_thresh).astype(int),
    )
    return (
        season.groupby(["county_name", "year"])
        .agg(
            {
                "clear_day": "sum",
                "anomaly_prcp": "sum",
                "anomaly_tmax": "sum",
                "anomaly_tmin": "sum",
                "anomaly_snow": "sum",
            }
        )
        .reset_index()
        .rename(
            columns={
                "clear_day": f"{season_name}_clear_day_count",
                "anomaly_prcp": f"{season_name}_anomaly_prcp_count",
                "anomaly_tmax": f"{season_name}_anomaly_tmax_count",
                "anomaly_tmin": f"{season_name}_anomaly_tmin_count",
                "anomaly_snow": f"{season_name}_anomaly_snow_count",
            }
        )
    )


def make_seasonal_features(df, config):
    """
    Build seasonal aggregates and anomaly/clear-day counts from config.
    config: dict with clear_thresh, anomaly_percentile, spring, summer, fall.
    Each season is (start_month, start_day, end_month, end_day).
    """
    clear_thresh = config.get("clear_thresh", 2.69)
    anomaly_percentile = config.get("anomaly_percentile") or {
        "prcp": 80, "tmax": 98, "tmin": 94, "snow": 93
    }
    spring = config.get("spring", (3, 13, 5, 28))
    summer = config.get("summer", (7, 6, 8, 31))
    fall = config.get("fall", (11, 4, 11, 29))

    s_spring = compute_seasonal_features(df, "spring", *spring)
    s_summer = compute_seasonal_features(df, "summer", *summer)
    s_fall = compute_seasonal_features(df, "fall", *fall)
    p_spring = compute_precipitation_features(
        df, "spring", *spring, clear_thresh=clear_thresh, anomaly_percentile=anomaly_percentile
    )
    p_summer = compute_precipitation_features(
        df, "summer", *summer, clear_thresh=clear_thresh, anomaly_percentile=anomaly_percentile
    )
    p_fall = compute_precipitation_features(
        df, "fall", *fall, clear_thresh=clear_thresh, anomaly_percentile=anomaly_percentile
    )

    spring_all = s_spring.merge(p_spring, on=["county_name", "year"], how="inner")
    summer_all = s_summer.merge(p_summer, on=["county_name", "year"], how="inner")
    fall_all = s_fall.merge(p_fall, on=["county_name", "year"], how="inner")
    seasonal_all = spring_all.merge(summer_all, on=["county_name", "year"], how="inner")
    seasonal_all = seasonal_all.merge(fall_all, on=["county_name", "year"], how="inner")
    return seasonal_all


def build_features(df, config, add_pca=False, n_pca_components=0):
    """
    Build feature matrix and target from cluster dataframe and config.
    Returns (X, y) or (X_df, y) with county_name, year preserved in index if needed.
    add_pca: if True, append PCA components (optional).
    """
    seasonal_all = make_seasonal_features(df, config)
    targets = df[["county_name", "year", "detrended_yield"]].drop_duplicates(
        subset=["county_name", "year"]
    )
    seasonal_all = pd.merge(seasonal_all, targets, on=["county_name", "year"], how="inner").dropna()

    if add_pca and n_pca_components > 0:
        import numpy as np
        from sklearn.decomposition import PCA
        from sklearn.impute import IterativeImputer
        from sklearn.experimental import enable_iterative_imputer

        numeric_cols = seasonal_all.select_dtypes(include=[np.number]).drop(
            columns=["detrended_yield"], errors="ignore"
        )
        imputer = IterativeImputer(random_state=42, max_iter=10)
        numeric_imputed = imputer.fit_transform(numeric_cols)
        pca = PCA(n_components=n_pca_components)
        components = pca.fit_transform(numeric_imputed)
        for i in range(n_pca_components):
            seasonal_all[f"pca_{i+1}"] = components[:, i]

    X = seasonal_all.drop(columns=["county_name", "year", "detrended_yield"])
    y = seasonal_all["detrended_yield"]
    return X, y
