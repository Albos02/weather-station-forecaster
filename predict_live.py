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

    # 3. Perform n iterative predictions
    n = 20
    predictions = []

    # Load pre-trained model
    model_path = "models/forecaster-30min.ubj"
    print(f"Loading pre-trained model from {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        return
    model = xgb.Booster()
    model.load_model(model_path)

    current_df = df.copy()

    for i in range(n):
        print(f"\n--- Iteration {i + 1} ---")

        # Feature Engineering for current data
        # Handle NaN values
        cols_num = current_df.select_dtypes(include=[np.number]).columns
        current_df[cols_num] = current_df[cols_num].ffill()

        # Add time features
        current_df[HOUR_COLUMN] = current_df["horodatage_référence"].dt.hour
        current_df[HOUR_SIN_COLUMN] = np.sin(2 * np.pi * current_df[HOUR_COLUMN] / 24)
        current_df[HOUR_COS_COLUMN] = np.cos(2 * np.pi * current_df[HOUR_COLUMN] / 24)
        current_df[WIND_DIR_SIN_COLUMN] = np.sin(
            2 * np.pi * current_df[WIND_DIRECTION_COLUMN] / 360
        )
        current_df[WIND_DIR_COS_COLUMN] = np.cos(
            2 * np.pi * current_df[WIND_DIRECTION_COLUMN] / 360
        )
        current_df["month"] = current_df["horodatage_référence"].dt.month
        current_df[MONTH_SIN_COLUMN] = np.sin(2 * np.pi * current_df["month"] / 12)
        current_df[MONTH_COS_COLUMN] = np.cos(2 * np.pi * current_df["month"] / 12)
        current_df[YEAR_DAY_PCT_COLUMN] = (
            current_df["horodatage_référence"].dt.dayofyear / 365
        )

        # Lags and Trends
        target = TARGET_COLUMN
        for shift in SHIFT_VALUES:
            minutes = shift * 10
            current_df[LAG_PREFIX.format(minutes)] = current_df[target].shift(shift)
            current_df[TREND_PREFIX.format(minutes)] = (
                current_df[target] - current_df[LAG_PREFIX.format(minutes)]
            )

        current_df[PRESSURE_6H_AGO_COLUMN] = current_df[PRESSURE_COLUMN].shift(
            PRESSURE_SHIFT
        )
        current_df[PRESSURE_TREND_6H_COLUMN] = (
            current_df[PRESSURE_COLUMN] - current_df[PRESSURE_6H_AGO_COLUMN]
        )

        # Feature selection
        features = (
            [LAG_PREFIX.format(s * 10) for s in SHIFT_VALUES]
            + [TREND_PREFIX.format(s * 10) for s in SHIFT_VALUES]
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
        features = [f for f in features if f in current_df.columns]

        # Get latest row
        X_live = current_df[features].iloc[[-1]]
        dlive = xgb.DMatrix(X_live, enable_categorical=True)

        y_pred = model.predict(dlive)
        current_wind = current_df[target].iloc[-1]

        if USE_DELTA_TARGET:
            prediction = y_pred[0] + current_wind
        elif USE_LOG_TARGET:
            prediction = np.expm1(y_pred[0])
        else:
            prediction = y_pred[0]

        pred_time = current_df["horodatage_référence"].iloc[-1] + pd.Timedelta(
            minutes=0
        )
        predictions.append((pred_time, prediction))
        print(f"Predicted wind at {pred_time}: {prediction:.2f} km/h")

        # Remove last row for next iteration
        current_df = current_df.iloc[:-1].reset_index(drop=True)

    # Visualization
    if "1" in show_graphs:
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))

            # Plot historical
            plt.plot(
                df["horodatage_référence"].tail(30),
                df[TARGET_COLUMN].tail(30),
                label="Actual Data",
                marker="o",
                color="blue",
            )

            # Plot predictions
            pred_times = [p[0] for p in predictions]
            pred_vals = [p[1] for p in predictions]
            plt.plot(
                pred_times,
                pred_vals,
                label="Predictions",
                marker="s",
                color="red",
                linestyle="--",
            )

            plt.title("Actual vs 3x Iterative Predictions")
            plt.xlabel("Time")
            plt.ylabel("Wind Speed (km/h)")
            plt.legend()
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
