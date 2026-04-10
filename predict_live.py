import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import os
import urllib.request
import ssl

# ============================================================
# CONFIGURATION CONSTANTS - Must match the training configuration
# ============================================================

# --- Data ---
DATA_FILE = "data/ogd-smn_bou_t_now.csv"  # Live data file
REMOTE_DATA_URL = (
    "https://data.geo.admin.ch/ch.meteoschweiz.ogd-smn/bou/ogd-smn_bou_t_now.csv"
)

# --- Data ---
DATA_FILE = "data/ogd-smn_bou_t_now.csv"  # Live data file
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

# --- Model Hyperparameters (must match training) ---
NUM_BOOST_ROUNDS = 7
MODEL_MAX_DEPTH = 5
MODEL_ETA = 0.1
RANDOM_STATE = 42
USE_GPU = True


def main():
    show_graphs = sys.argv[1] if len(sys.argv) > 1 else ""

    # 1. Download latest data from remote source
    print(f"Downloading latest data from {REMOTE_DATA_URL}...")
    try:
        # Download the file
        urllib.request.urlretrieve(REMOTE_DATA_URL, DATA_FILE)
        print("Download completed successfully.")
    except Exception as e:
        print(f"Warning: Failed to download data ({e}). Using local file if available.")

    # 2. Load and sort data from CSV
    print("Loading live CSV data...")
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

    print(f"Loaded {len(df)} rows of data")
    print(f"Latest timestamp: {df['horodatage_référence'].iloc[-1]}")

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

    # For live prediction, we only need the latest row that has all features
    # We'll keep all rows for now but note that the last few rows may have NaN
    df_clean = df.dropna(subset=features_to_check)

    if len(df_clean) == 0:
        print(
            "Error: Not enough data to create features. Need at least {} rows.".format(
                max(SHIFT_VALUES) + TARGET_DISTANCE + PRESSURE_SHIFT
            )
        )
        return

    print(f"After cleaning, {len(df_clean)} rows available for feature creation")

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

    # Ensure columns exist before modeling
    features = [f for f in features if f in df_clean.columns]

    # For live prediction, we want to predict the next value
    # So we use the last row that has all features
    last_row_idx = len(df_clean) - 1
    X_live = df_clean[features].iloc[[last_row_idx]]
    current_wind = df_clean[target].iloc[last_row_idx]

    print(
        f"Making prediction based on data from: {df_clean['horodatage_référence'].iloc[last_row_idx]}"
    )
    print(f"Current wind speed: {current_wind:.2f} km/h")

    # Prepare data in DMatrix format (GPU optimized)
    dlive = xgb.DMatrix(X_live, enable_categorical=True)

    # Load pre-trained model
    model_path = "models/forecaster-30min.ubj"
    print(f"Loading pre-trained model from {model_path}")

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return

    model = xgb.Booster()
    model.load_model(model_path)

    # Prediction
    print("Making live prediction...")
    y_pred = model.predict(dlive)

    # Convert delta prediction back to absolute value if needed
    if USE_DELTA_TARGET:
        prediction = y_pred[0] + current_wind
    elif USE_LOG_TARGET:
        prediction = np.expm1(y_pred[0])
    else:
        prediction = y_pred[0]

    print(f"\n=== LIVE PREDICTION ===")
    print(f"Predicted wind speed in 30 minutes: {prediction:.2f} km/h")
    print(f"Predicted change: {y_pred[0]:.2f} km/h")

    # Optional: Show recent trend
    if len(df_clean) >= 6:
        recent_winds = df_clean[target].tail(6).values
        print(f"\nRecent wind speeds (last 6 observations):")
        for i, ws in enumerate(recent_winds):
            print(f"  T-{5 - i * 10}min: {ws:.2f} km/h")

    # Optional: Show temporal plot
    if "1" in show_graphs and len(df_clean) >= 10:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))

            # Plot recent history
            recent_history = df_clean.tail(20)
            plt.plot(
                recent_history["horodatage_référence"],
                recent_history[target],
                label="Historical",
                color="blue",
                marker="o",
            )

            # Plot prediction point
            future_time = recent_history["horodatage_référence"].iloc[
                -1
            ] + pd.Timedelta(minutes=30)
            plt.plot(
                [future_time],
                [prediction],
                label="Prediction (30min ahead)",
                color="red",
                marker="s",
                markersize=10,
            )

            plt.title("Wind Speed Prediction - Live Update")
            plt.xlabel("Time")
            plt.ylabel("Wind Speed (km/h)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available for plotting")


if __name__ == "__main__":
    main()
