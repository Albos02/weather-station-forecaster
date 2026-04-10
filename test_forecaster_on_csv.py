import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import time, os
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    max_error,
    r2_score,
    mean_squared_error,
)

# ============================================================
# CONFIGURATION CONSTANTS - Tweak these to adjust behavior
# ============================================================

# --- Data ---
DATA_FILE = "data/ogd-smn_bou_t_recent.csv"  # Changed to T file (10-min intervals)
TARGET_COLUMN = "vitesse_vent_moyenne_10min_kmh"
PRESSURE_COLUMN = "pression_barométrique_qfe"
WIND_DIRECTION_COLUMN = "direction_du_vent_moyenne_10min"
WIND_DIR_SIN_COLUMN = "wind_dir_sin"
WIND_DIR_COS_COLUMN = "wind_dir_cos"
HUMIDITY_COLUMN = "humidité"
AIR_TEMP_COLUMN = "température_air"
GUST_COLUMN = "rafale_3s_maximum_kmh"

# --- Target ---
TARGET_DISTANCE = 3  # * 10 = prediction horizon in minutes (3 = 30min ahead)
TARGET_COLUMN_NAME = "target_30min"
USE_DELTA_TARGET = True  # Predict change in wind speed instead of absolute value
USE_LOG_TARGET = False  # Log-transform target for relative changes

# --- Lag Features ---
SHIFT_VALUES = [1, 2, 3, 4, 5, 6]  # Number of 10-min steps back for lag features
LAG_PREFIX = "wind_{}min_ago"
TREND_PREFIX = "wind_trend_{}min"

# --- Pressure ---
PRESSURE_SHIFT = 36  # Number of 10-min steps back for pressure (36 = 6h ago)
PRESSURE_6H_AGO_COLUMN = "pressure_6h_ago"
PRESSURE_TREND_6H_COLUMN = "pressure_trend_6h"

# --- Temporal Features ---
HOUR_COLUMN = "hour"
HOUR_SIN_COLUMN = "hour_sin"
HOUR_COS_COLUMN = "hour_cos"
MONTH_SIN_COLUMN = "month_sin"
MONTH_COS_COLUMN = "month_cos"
YEAR_DAY_PCT_COLUMN = "year_day_pct"

# --- Train/Test Split ---
TRAIN_SPLIT_RATIO = 0.8
TEST_PRED_SLICE = TARGET_DISTANCE  # How many rows to slice off for alignment

# --- Model Hyperparameters ---
NUM_BOOST_ROUNDS = 7
MODEL_MAX_DEPTH = 5
MODEL_ETA = 0.1
RANDOM_STATE = 42
USE_GPU = True

# --- Evaluation Thresholds ---
ERROR_THRESHOLDS = [1, 2, 5, 10]  # km/h thresholds for error percentage reporting

# --- Loss Configuration ---
USE_HUBER_LOSS = False  # Use Huber loss (robust to outliers)
HUBER_DELTA = 1.0  # Threshold for Huber loss (transition from quadratic to linear)
USE_PINBALL_LOSS = True  # Use pinball loss (quantile loss)
PINBALL_QUANTILE = 0.5  # Quantile for pinball loss (0.5 = median)

LOSS_CONFIGS = [
    {
        "name": "peak",
        "enabled": False,
        "threshold": 18.0,
        "weight_factor": 1.0,
        "direction": "above",
    },
    {
        "name": "low_wind",
        "enabled": False,
        "threshold": 5.0,
        "weight_factor": 1.5,
        "direction": "below",
    },
]
LOSS_COMBINATION = "additive"
LOSS_EVAL_METRIC_NAME = "combined_rmse"

# ============================================================


def main():
    show_graphs = sys.argv[1] if len(sys.argv) > 1 else ""

    # 1. Load and sort data from CSV
    print("Loading CSV data...")
    df = pd.read_csv(DATA_FILE, sep=";")

    # Rename columns to match expected names (T file format)
    column_mapping = {
        "reference_timestamp": "horodatage_référence",
        "fkl010z0": "vitesse_vent_moyenne_10min_kmh",  # wind speed
        "dkl010z0": "direction_du_vent_moyenne_10min",  # wind direction
        "ure200s0": "humidité",  # humidity
        "tre200s0": "température_air",  # temperature
        "prestas0": "pression_barométrique_qfe",  # pressure
        "fu3010z0": "rafale_3s_maximum_kmh",  # gust
    }

    # Only rename columns that exist
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Convert timestamp
    df["horodatage_référence"] = pd.to_datetime(
        df["horodatage_référence"], format="%d.%m.%Y %H:%M"
    )
    df = df.sort_values("horodatage_référence").reset_index(drop=True)

    # 2. Handle NaN values - causal forward fill only (no backward fill to prevent leakage)
    cols_num = df.select_dtypes(include=[np.number]).columns
    df[cols_num] = df[cols_num].ffill()

    # 3. Feature Engineering - Hour, Month and Time Lags
    df[HOUR_COLUMN] = df["horodatage_référence"].dt.hour
    df[HOUR_SIN_COLUMN] = np.sin(2 * np.pi * df[HOUR_COLUMN] / 24)
    df[HOUR_COS_COLUMN] = np.cos(2 * np.pi * df[HOUR_COLUMN] / 24)
    df[WIND_DIR_SIN_COLUMN] = np.sin(2 * np.pi * df[WIND_DIRECTION_COLUMN] / 360)
    df[WIND_DIR_COS_COLUMN] = np.cos(2 * np.pi * df[WIND_DIRECTION_COLUMN] / 360)
    df["month"] = df["horodatage_référence"].dt.month
    df[MONTH_SIN_COLUMN] = np.sin(2 * np.pi * df["month"] / 12)
    df[MONTH_COS_COLUMN] = np.cos(2 * np.pi * df["month"] / 12)
    df[YEAR_DAY_PCT_COLUMN] = df["horodatage_référence"].dt.dayofyear / 365

    target = TARGET_COLUMN

    df[TARGET_COLUMN_NAME] = df[target].shift(-TARGET_DISTANCE)

    if USE_DELTA_TARGET:
        df["target_30min_delta"] = df[TARGET_COLUMN_NAME] - df[target]
        target_col = "target_30min_delta"
    elif USE_LOG_TARGET:
        df["target_30min_log"] = np.log1p(df[TARGET_COLUMN_NAME])
        target_col = "target_30min_log"
    else:
        target_col = TARGET_COLUMN_NAME

    for shift in SHIFT_VALUES:
        minutes = shift * 10
        df[LAG_PREFIX.format(minutes)] = df[target].shift(shift)

    for shift in SHIFT_VALUES:
        minutes = shift * 10
        df[TREND_PREFIX.format(minutes)] = df[target] - df[LAG_PREFIX.format(minutes)]

    df[PRESSURE_6H_AGO_COLUMN] = df[PRESSURE_COLUMN].shift(PRESSURE_SHIFT)
    df[PRESSURE_TREND_6H_COLUMN] = df[PRESSURE_COLUMN] - df[PRESSURE_6H_AGO_COLUMN]

    # Drop rows with NaN in features used
    lag_cols = [LAG_PREFIX.format(s * 10) for s in SHIFT_VALUES]
    trend_cols = [TREND_PREFIX.format(s * 10) for s in SHIFT_VALUES]

    features_to_check = (
        lag_cols
        + trend_cols
        + [
            WIND_DIR_SIN_COLUMN,
            WIND_DIR_COS_COLUMN,
            HOUR_SIN_COLUMN,
            HOUR_COS_COLUMN,
            MONTH_SIN_COLUMN,
            MONTH_COS_COLUMN,
            YEAR_DAY_PCT_COLUMN,
            PRESSURE_TREND_6H_COLUMN,
            HUMIDITY_COLUMN,
            AIR_TEMP_COLUMN,
            GUST_COLUMN,
        ]
    )
    df = df.dropna(subset=features_to_check)

    # Feature selection
    features = (
        lag_cols
        + trend_cols
        + [
            WIND_DIR_SIN_COLUMN,
            WIND_DIR_COS_COLUMN,
            HOUR_SIN_COLUMN,
            HOUR_COS_COLUMN,
            MONTH_SIN_COLUMN,
            MONTH_COS_COLUMN,
            YEAR_DAY_PCT_COLUMN,
            PRESSURE_TREND_6H_COLUMN,
            HUMIDITY_COLUMN,
            AIR_TEMP_COLUMN,
            GUST_COLUMN,
        ]
    )

    # Ensure columns exist before split
    features = [f for f in features if f in df.columns]

    # Chronological split (past for training)
    split_idx = int(len(df) * TRAIN_SPLIT_RATIO)
    train_df = df.iloc[: split_idx - TARGET_DISTANCE]
    test_df = df.iloc[split_idx:]

    X_train, y_train = train_df[features], train_df[target_col]
    X_test, y_test = test_df[features], test_df[target_col]

    train_mask = y_train.notna()
    X_train, y_train = X_train[train_mask], y_train[train_mask]

    test_mask = y_test.notna()
    X_test, y_test = X_test[test_mask], y_test[test_mask]

    # Prepare data in DMatrix format (GPU optimized)
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # Load pre-trained model
    model_path = "models/forecaster-30min.ubj"
    print(f"Loading pre-trained model from {model_path}")
    model = xgb.Booster()
    model.load_model(model_path)

    # Prediction
    print("Making predictions...")
    y_pred = model.predict(dtest)
    y_pred_train = model.predict(dtrain)

    # Capture variance of deltas for R2 history calculation before converting to absolutes
    var_y_test_delta = np.var(y_test)
    var_y_train_delta = np.var(y_train)

    # Convert delta predictions back to absolute values for evaluation
    if USE_DELTA_TARGET:
        current_wind_test = test_df[target].loc[y_test.index].values
        current_wind_train = train_df[target].iloc[: len(y_pred_train)].values
        y_test = y_test + current_wind_test
        y_pred = y_pred + current_wind_test
        y_train = y_train + current_wind_train
        y_pred_train = y_pred_train + current_wind_train

    elif USE_LOG_TARGET:
        y_test = np.expm1(y_test)
        y_pred = np.expm1(y_pred)
        y_pred_train = np.expm1(y_pred_train)

    y_test = y_test[:-TEST_PRED_SLICE]
    y_pred = y_pred[TEST_PRED_SLICE:]

    # Evaluation
    print("\n--- RESULTS ---")
    mae = mean_absolute_error(y_test, y_pred)
    medianae = median_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    r2_train = r2_score(y_train, y_pred_train)
    max_er = max_error(y_test, y_pred)
    abs_errors = np.abs(y_test - y_pred)
    error_pcts = {t: np.mean(abs_errors > t) * 100 for t in ERROR_THRESHOLDS}

    print(f"R2 Score (TEST): {r2:.4f}")
    print(f"R2 Score (TRAIN): {r2_train:.4f} (check for overfitting)")
    print(f"Mean Absolute Error (MAE): {mae:.2f} km/h")
    print(f"Median Absolute Error (MedianAE): {medianae:.2f} km/h")
    print(f"RMSE: {rmse:.2f} km/h")
    print(f"Max Error: {max_er:.2f} km/h")
    for t in ERROR_THRESHOLDS:
        print(f"Errors > {t} km/h: {error_pcts[t]:.2f}%")

    # Calculate R2 per iteration (using training metrics from original model training)
    # Since we're using a pre-trained model, we don't have the training history
    # We'll just report the final R2

    print("\n--- Naivity Score ---")
    first_lag_col = LAG_PREFIX.format(SHIFT_VALUES[0] * 10)
    y_pred_naiv = test_df.loc[y_test.index, first_lag_col].values

    print(f"Naive Model R2 Score: {r2_score(y_test, y_pred_naiv):.4f}")
    print(
        f"Naive Model Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_naiv):.2f} km/h"
    )

    # Optional: Show temporal plot
    if "1" in show_graphs:
        plt.figure(figsize=(18, 6))
        plt.plot(
            y_test.values[:300],
            label="Actual",
            color="black",
            alpha=0.7,
        )
        plt.plot(
            y_pred[:300],
            label="Predicted",
            color="orange",
            linestyle="--",
        )
        plt.title("Actual vs Predicted Wind Speed (First 300 test points)")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
