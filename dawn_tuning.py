"""
Tuning utilities for corn yield models: season date ranges (binary + random search)
and hyperparameters (clear_thresh, anomaly_percentile).
Used by dawn_models_clean.ipynb — run tuning before or alongside model training.
"""
import random
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from dawn_features import build_features


def _evaluate_config(df, config):
    """Build features from config, train RF, return test R². Returns -np.inf on failure."""
    try:
        X, y = build_features(df, config)
        if len(X) < 20:
            return -np.inf
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        return r2_score(y_test, rf.predict(X_test))
    except Exception:
        return -np.inf


# -----------------------------------------------------------------------------
# Binary search: optimize start/end dates per season (spring → summer → fall)
# -----------------------------------------------------------------------------

def _days_to_month_day(days_offset, base_date):
    d = base_date + timedelta(days=days_offset)
    return (d.month, d.day)


def date_range_binary_search_spring(
    df,
    summer=(6, 1, 8, 31),
    fall=(9, 1, 11, 30),
    clear_thresh=2.0,
    anomaly_percentile=None,
    verbose=True,
):
    """
    Optimize spring (start_month, start_day, end_month, end_day) using binary search.
    summer and fall are fixed as (sm, sd, em, ed). Returns ((s_start, s_end), best_r2).
    """
    if anomaly_percentile is None:
        anomaly_percentile = {"prcp": 77, "tmax": 95, "tmin": 91, "snow": 84}
    base = date(2025, 3, 1)
    start_min, start_max = 0, (date(2025, 6, 1) - base).days
    end_min, end_max = 14, (date(2025, 6, 30) - base).days

    best_start_offset = (date(2025, 3, 15) - base).days
    best_end_offset = (date(2025, 5, 5) - base).days
    best_r2 = -999.0

    for step in [14, 7, 3, 1]:
        improved = True
        while improved:
            improved = False
            for delta in [-step, step]:
                trial_start = int(np.clip(best_start_offset + delta, start_min, best_end_offset - 1))
                s_start = _days_to_month_day(trial_start, base)
                s_end = _days_to_month_day(best_end_offset, base)
                spring = (s_start[0], s_start[1], s_end[0], s_end[1])
                config = {
                    "clear_thresh": clear_thresh,
                    "anomaly_percentile": anomaly_percentile,
                    "spring": spring,
                    "summer": summer,
                    "fall": fall,
                }
                r2 = _evaluate_config(df, config)
                if r2 > best_r2:
                    best_start_offset = int(trial_start)
                    best_r2 = r2
                    improved = True
                    if verbose:
                        print(f"  New best spring start: {_days_to_month_day(best_start_offset, base)} to {_days_to_month_day(best_end_offset, base)} R²={best_r2:.4f}")

        improved = True
        while improved:
            improved = False
            for delta in [-step, step]:
                trial_end = int(np.clip(best_end_offset + delta, best_start_offset + 1, end_max))
                s_start = _days_to_month_day(best_start_offset, base)
                s_end = _days_to_month_day(trial_end, base)
                spring = (s_start[0], s_start[1], s_end[0], s_end[1])
                config = {
                    "clear_thresh": clear_thresh,
                    "anomaly_percentile": anomaly_percentile,
                    "spring": spring,
                    "summer": summer,
                    "fall": fall,
                }
                r2 = _evaluate_config(df, config)
                if r2 > best_r2:
                    best_end_offset = trial_end
                    best_r2 = r2
                    improved = True
                    if verbose:
                        print(f"  New best spring end: {_days_to_month_day(best_start_offset, base)} to {_days_to_month_day(best_end_offset, base)} R²={best_r2:.4f}")

    s_start = _days_to_month_day(best_start_offset, base)
    s_end = _days_to_month_day(best_end_offset, base)
    spring_range = (s_start[0], s_start[1], s_end[0], s_end[1])
    if verbose:
        print(f"  Final optimal spring: {spring_range[0]},{spring_range[1]} to {spring_range[2]},{spring_range[3]} R²={best_r2:.4f}")
    return spring_range, best_r2


def date_range_binary_search_summer(
    df,
    spring,
    fall=(9, 1, 11, 30),
    clear_thresh=2.0,
    anomaly_percentile=None,
    verbose=True,
):
    """Optimize summer (sm, sd, em, ed). spring is (sm, sd, em, ed). Returns (summer_range, best_r2)."""
    if anomaly_percentile is None:
        anomaly_percentile = {"prcp": 77, "tmax": 95, "tmin": 91, "snow": 84}
    base = date(2025, 5, 1)
    start_min, end_max = 0, (date(2025, 9, 30) - base).days
    best_start_offset = 0
    best_end_offset = (date(2025, 8, 31) - base).days
    best_r2 = -999.0

    for step in [14, 7, 3, 1]:
        improved = True
        while improved:
            improved = False
            for delta in [-step, step]:
                trial_start = np.clip(best_start_offset + delta, start_min, best_end_offset - 1)
                summer = _days_to_month_day(trial_start, base) + _days_to_month_day(best_end_offset, base)
                config = {
                    "clear_thresh": clear_thresh,
                    "anomaly_percentile": anomaly_percentile,
                    "spring": spring,
                    "summer": summer,
                    "fall": fall,
                }
                r2 = _evaluate_config(df, config)
                if r2 > best_r2:
                    best_start_offset = int(trial_start)
                    best_r2 = r2
                    improved = True
                    if verbose:
                        print(f"  New best summer start: {summer} R²={best_r2:.4f}")

        improved = True
        while improved:
            improved = False
            for delta in [-step, step]:
                trial_end = np.clip(best_end_offset + delta, best_start_offset + 1, end_max)
                summer = _days_to_month_day(best_start_offset, base) + _days_to_month_day(trial_end, base)
                config = {
                    "clear_thresh": clear_thresh,
                    "anomaly_percentile": anomaly_percentile,
                    "spring": spring,
                    "summer": summer,
                    "fall": fall,
                }
                r2 = _evaluate_config(df, config)
                if r2 > best_r2:
                    best_end_offset = int(trial_end)
                    best_r2 = r2
                    improved = True
                    if verbose:
                        print(f"  New best summer end: {summer} R²={best_r2:.4f}")

    summer_range = _days_to_month_day(best_start_offset, base) + _days_to_month_day(best_end_offset, base)
    if verbose:
        print(f"  Final optimal summer: {summer_range} R²={best_r2:.4f}")
    return summer_range, best_r2


def date_range_binary_search_fall(
    df,
    spring,
    summer,
    clear_thresh=2.0,
    anomaly_percentile=None,
    verbose=True,
):
    """Optimize fall (sm, sd, em, ed). Returns (fall_range, best_r2)."""
    if anomaly_percentile is None:
        anomaly_percentile = {"prcp": 77, "tmax": 95, "tmin": 91, "snow": 84}
    base = date(2025, 8, 15)
    start_min, end_max = 0, (date(2025, 11, 30) - base).days
    best_start_offset = 0
    best_end_offset = (date(2025, 11, 30) - base).days
    best_r2 = -999.0

    for step in [14, 7, 3, 1]:
        improved = True
        while improved:
            improved = False
            for delta in [-step, step]:
                trial_start = np.clip(best_start_offset + delta, start_min, best_end_offset - 1)
                fall = _days_to_month_day(trial_start, base) + _days_to_month_day(best_end_offset, base)
                config = {
                    "clear_thresh": clear_thresh,
                    "anomaly_percentile": anomaly_percentile,
                    "spring": spring,
                    "summer": summer,
                    "fall": fall,
                }
                r2 = _evaluate_config(df, config)
                if r2 > best_r2:
                    best_start_offset = int(trial_start)
                    best_r2 = r2
                    improved = True
                    if verbose:
                        print(f"  New best fall start: {fall} R²={best_r2:.4f}")

        improved = True
        while improved:
            improved = False
            for delta in [-step, step]:
                trial_end = np.clip(best_end_offset + delta, best_start_offset + 1, end_max)
                fall = _days_to_month_day(best_start_offset, base) + _days_to_month_day(trial_end, base)
                config = {
                    "clear_thresh": clear_thresh,
                    "anomaly_percentile": anomaly_percentile,
                    "spring": spring,
                    "summer": summer,
                    "fall": fall,
                }
                r2 = _evaluate_config(df, config)
                if r2 > best_r2:
                    best_end_offset = int(trial_end)
                    best_r2 = r2
                    improved = True
                    if verbose:
                        print(f"  New best fall end: {fall} R²={best_r2:.4f}")

    fall_range = _days_to_month_day(best_start_offset, base) + _days_to_month_day(best_end_offset, base)
    if verbose:
        print(f"  Final optimal fall: {fall_range} R²={best_r2:.4f}")
    return fall_range, best_r2


def run_binary_search_all_seasons(df, clear_thresh=2.0, anomaly_percentile=None, verbose=True):
    """
    Run binary search for spring, then summer (given spring), then fall (given spring+summer).
    Returns full config dict with spring, summer, fall, clear_thresh, anomaly_percentile.
    """
    if anomaly_percentile is None:
        anomaly_percentile = {"prcp": 77, "tmax": 95, "tmin": 91, "snow": 84}
    default_summer, default_fall = (6, 1, 8, 31), (9, 1, 11, 30)

    if verbose:
        print("Binary search: Spring")
    spring, _ = date_range_binary_search_spring(
        df, summer=default_summer, fall=default_fall,
        clear_thresh=clear_thresh, anomaly_percentile=anomaly_percentile, verbose=verbose
    )
    if verbose:
        print("Binary search: Summer")
    summer, _ = date_range_binary_search_summer(
        df, spring=spring, fall=default_fall,
        clear_thresh=clear_thresh, anomaly_percentile=anomaly_percentile, verbose=verbose
    )
    if verbose:
        print("Binary search: Fall")
    fall, best_r2 = date_range_binary_search_fall(
        df, spring=spring, summer=summer,
        clear_thresh=clear_thresh, anomaly_percentile=anomaly_percentile, verbose=verbose
    )
    return {
        "spring": spring,
        "summer": summer,
        "fall": fall,
        "clear_thresh": clear_thresh,
        "anomaly_percentile": anomaly_percentile,
    }, best_r2


# -----------------------------------------------------------------------------
# Random search: season date ranges
# -----------------------------------------------------------------------------

def random_search_optimize_dates(
    df,
    num_iterations=100,
    clear_thresh=2.0,
    anomaly_percentile=None,
    verbose=True,
):
    """
    Random search over (spring, summer, fall) date ranges. Each season is (sm, sd, em, ed).
    Returns (best_config, best_r2, results_list).
    """
    if anomaly_percentile is None:
        anomaly_percentile = {"prcp": 77, "tmax": 95, "tmin": 91, "snow": 84}

    def random_date_range(start_day, end_day):
        delta_days = (end_day - start_day).days
        if delta_days < 20:
            raise ValueError("Date range too short")
        s_offset = random.randint(0, max(0, delta_days - 15))
        e_offset = random.randint(s_offset + 10, delta_days)
        s = start_day + timedelta(days=s_offset)
        e = start_day + timedelta(days=e_offset)
        return (s.month, s.day, e.month, e.day)

    spring_base = date(2025, 3, 1)
    summer_base = date(2025, 5, 15)
    fall_base = date(2025, 8, 15)

    best_config = None
    best_r2 = -np.inf
    results = []
    seen = set()

    for _ in range(num_iterations):
        try:
            spring_range = random_date_range(spring_base, date(2025, 5, 31))
            summer_range = random_date_range(summer_base, date(2025, 8, 31))
            fall_range = random_date_range(fall_base, date(2025, 11, 30))
        except ValueError:
            continue
        key = (spring_range, summer_range, fall_range)
        if key in seen:
            continue
        seen.add(key)

        config = {
            "clear_thresh": clear_thresh,
            "anomaly_percentile": anomaly_percentile,
            "spring": spring_range,
            "summer": summer_range,
            "fall": fall_range,
        }
        r2 = _evaluate_config(df, config)
        results.append((r2, spring_range, summer_range, fall_range))

        if r2 > best_r2:
            best_r2 = r2
            best_config = config
            if verbose:
                print(f"  New best R²: {r2:.4f}  Spring: {spring_range}, Summer: {summer_range}, Fall: {fall_range}")

    if verbose and best_config:
        print(f"  Best R²: {best_r2:.4f}")
    return best_config, best_r2, results


# -----------------------------------------------------------------------------
# Hyperparameter tuning: clear_thresh + anomaly_percentile
# -----------------------------------------------------------------------------

def tune_hyperparameters(
    df,
    date_config=None,
    clear_thresh_values=(1.0, 2.0, 2.5, 3.0, 3.5, 5.0),
    anomaly_percentile_grid=None,
    num_random_samples=50,
    use_random=True,
    verbose=True,
):
    """
    Tune clear_thresh and anomaly_percentile (prcp, tmax, tmin, snow).
    date_config: dict with spring, summer, fall (and optionally clear_thresh, anomaly_percentile).
      If None, uses default fixed seasons.
    If use_random=True, samples num_random_samples from the grid; else does full grid (can be slow).
    Returns (best_config, best_r2).
    """
    if date_config is None:
        date_config = {
            "spring": (3, 13, 5, 28),
            "summer": (7, 6, 8, 31),
            "fall": (11, 4, 11, 29),
        }
    base_config = {
        "spring": date_config.get("spring", (3, 13, 5, 28)),
        "summer": date_config.get("summer", (7, 6, 8, 31)),
        "fall": date_config.get("fall", (11, 4, 11, 29)),
    }
    if anomaly_percentile_grid is None:
        percentile_options = [70, 75, 80, 85, 90, 93, 95, 98, 99]
        if use_random:
            anomaly_percentile_grid = [
                {
                    "prcp": random.choice(percentile_options),
                    "tmax": random.choice(percentile_options),
                    "tmin": random.choice(percentile_options),
                    "snow": random.choice(percentile_options),
                }
                for _ in range(num_random_samples)
            ]
        else:
            anomaly_percentile_grid = [
                {"prcp": p1, "tmax": p2, "tmin": p3, "snow": p4}
                for p1 in percentile_options
                for p2 in percentile_options
                for p3 in percentile_options
                for p4 in percentile_options
            ]

    best_config = None
    best_r2 = -np.inf
    full_trials = [(ct, ap) for ct in clear_thresh_values for ap in anomaly_percentile_grid]
    if use_random and len(full_trials) > num_random_samples:
        trials = random.sample(full_trials, num_random_samples)
    else:
        trials = full_trials

    for clear_thresh, anomaly_percentile in trials:
        config = {**base_config, "clear_thresh": clear_thresh, "anomaly_percentile": anomaly_percentile}
        r2 = _evaluate_config(df, config)
        if r2 > best_r2:
            best_r2 = r2
            best_config = config
            if verbose:
                print(f"  New best R²: {r2:.4f}  clear_thresh={clear_thresh}, anomaly_percentile={anomaly_percentile}")

    if verbose and best_config:
        print(f"  Best clear_thresh: {best_config['clear_thresh']}, anomaly_percentile: {best_config['anomaly_percentile']}")
    return best_config, best_r2
