import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    max_error,
    r2_score,
    mean_squared_error,
)

import sys
import time, os
import xgboost as xgb

# ============================================================
# CONFIGURATION CONSTANTS - Tweak these to adjust behavior
# ============================================================

### Current Best Results :
# Delta Target = True
# Log Target = False
#
# Loss Config : Pinball Loss (quantile=0.5)
#
# Model max_depth=5
# num_boost_round=7
#
# Test R2: 0.9958
# MAE: 0.44 km/h
#


# --- Data ---
DATA_FILE = "data/processed_windspeed.parquet"
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

# --- Visualization ---
PLOT_FIGSIZE_MAIN = (18, 15)
PLOT_FIGSIZE_TEMPORAL = (18, 6)
PLOT_TEMPORAL_ZOOM_ROWS = 300  # Number of rows to show in temporal zoom
PLOT_ERROR_HIST_BINS = 100
PLOT_R2_FOCUS_LAST_PCT = 0.5  # Show last 50% of iterations for R2 convergence
PLOT_R2_Y_PADDING_LOW = 0.20  # 20% padding below
PLOT_R2_Y_PADDING_HIGH = 0.05  # 5% padding above
PLOT_R2_MAX_Y_CAP = 1.001  # Cap R2 y-axis at this value

# ============================================================

show_graphs = sys.argv[1] if len(sys.argv) > 1 else ""

# 1. Load and sort data
df = pd.read_parquet(DATA_FILE, engine="pyarrow")
df["horodatage_référence"] = pd.to_datetime(df["horodatage_référence"], dayfirst=True)
df = df.sort_values("horodatage_référence").reset_index(drop=True)

# 2. Handle NaN values - linear interpolation
cols_num = df.select_dtypes(include=[np.number]).columns
df[cols_num] = df[cols_num].interpolate(method="linear").bfill()

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
train_df = df.iloc[:split_idx]
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


### DYNAMIC WEIGHTED LOSS
# Custom loss: dynamically built from LOSS_CONFIGS


def _compute_weights(y_true):
    deltas = []
    for cfg in LOSS_CONFIGS:
        if not cfg["enabled"]:
            continue
        if cfg["direction"] == "above":
            delta = (
                cfg["weight_factor"]
                * np.maximum(0, y_true - cfg["threshold"])
                / cfg["threshold"]
            )
        else:
            delta = (
                cfg["weight_factor"]
                * np.maximum(0, cfg["threshold"] - y_true)
                / cfg["threshold"]
            )
        deltas.append(delta)

    if LOSS_COMBINATION == "additive":
        return 1.0 + sum(deltas)
    else:
        weights = np.ones_like(y_true)
        for d in deltas:
            weights *= 1.0 + d
        return weights


def huber_obj(preds, dtrain):
    y_true = dtrain.get_label()
    errors = preds - y_true
    abs_errors = np.abs(errors)
    grad = np.where(
        abs_errors <= HUBER_DELTA,
        -errors,
        -HUBER_DELTA * np.sign(errors),
    )
    hess = np.where(abs_errors <= HUBER_DELTA, 1.0, 1e-6)
    return grad, hess


def huber_eval(preds, dtrain):
    y_true = dtrain.get_label()
    errors = preds - y_true
    abs_errors = np.abs(errors)
    loss = np.mean(
        np.where(
            abs_errors <= HUBER_DELTA,
            errors**2,
            HUBER_DELTA * (2.0 * abs_errors - HUBER_DELTA),
        )
    )
    return "huber_rmse", np.sqrt(loss)


def pinball_obj(preds, dtrain):
    y_true = dtrain.get_label()
    errors = preds - y_true
    grad = np.where(errors > 0, PINBALL_QUANTILE, PINBALL_QUANTILE - 1)
    hess = np.ones_like(errors)
    return grad, hess


def pinball_eval(preds, dtrain):
    y_true = dtrain.get_label()
    errors = y_true - preds
    pinball = np.mean(
        np.where(errors > 0, PINBALL_QUANTILE * errors, (PINBALL_QUANTILE - 1) * errors)
    )
    return "pinball_loss", pinball


def dynamic_weighted_loss(preds, dtrain):
    y_true = dtrain.get_label()
    errors = preds - y_true
    weights = _compute_weights(y_true)
    grad = 2.0 * errors * weights
    hess = 2.0 * weights
    return grad, hess


def dynamic_weighted_eval(preds, dtrain):
    y_true = dtrain.get_label()
    errors = preds - y_true
    weights = _compute_weights(y_true)
    loss = np.mean(weights * errors**2)
    return LOSS_EVAL_METRIC_NAME, np.sqrt(loss)


# Model configuration for GPU
if USE_HUBER_LOSS:
    params = {
        "max_depth": MODEL_MAX_DEPTH,
        "eta": MODEL_ETA,
        "objective": "reg:pseudohubererror",
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
        "random_state": RANDOM_STATE,
        "huber_slope": HUBER_DELTA,
    }
else:
    params = {
        "max_depth": MODEL_MAX_DEPTH,
        "eta": MODEL_ETA,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": "cuda" if USE_GPU else "cpu",
        "random_state": RANDOM_STATE,
    }

evals_result = {}

print(f"Starting GPU training on {len(X_train)} rows...")
start_time = time.time()


def select_loss():
    if USE_HUBER_LOSS:
        return None  # Use built-in Huber
    if USE_PINBALL_LOSS:
        return pinball_obj
    enabled = [cfg for cfg in LOSS_CONFIGS if cfg["enabled"]]
    if len(enabled) == 0:
        return None
    return dynamic_weighted_loss


def select_eval():
    if USE_HUBER_LOSS:
        return huber_eval
    if USE_PINBALL_LOSS:
        return pinball_eval
    return dynamic_weighted_eval


model = xgb.train(
    params,
    dtrain,
    num_boost_round=NUM_BOOST_ROUNDS,
    obj=select_loss(),
    evals=[(dtrain, "train"), (dtest, "test")],
    evals_result=evals_result,
    verbose_eval=False,
)

end_time = time.time()
duration = end_time - start_time

print(f"Completed in {duration:.2f} seconds ({duration / 60:.2f} minutes)")

# Prediction
y_pred = model.predict(dtest)
y_pred_train = model.predict(dtrain)

# Convert delta predictions back to absolute values for evaluation
if USE_DELTA_TARGET:
    current_wind_test = test_df[target].loc[y_test.index].values
    current_wind_train = train_df[target].iloc[: len(y_pred_train)].values
    y_test = y_test + current_wind_test
    y_pred = y_pred + current_wind_test
    y_pred_train = y_pred_train + current_wind_train

elif USE_LOG_TARGET:
    y_test = np.expm1(y_test)
    y_pred = np.expm1(y_pred)
    y_pred_train = np.expm1(y_pred_train)

y_test = y_test[:-TEST_PRED_SLICE]
y_pred = y_pred[TEST_PRED_SLICE:]


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
medianae = median_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)
max_er = max_error(y_test, y_pred)
abs_errors = np.abs(y_test - y_pred)
error_pcts = {t: np.mean(abs_errors > t) * 100 for t in ERROR_THRESHOLDS}


print(f"--- RESULTS ---")
print(f"R2 Score (TEST): {r2:.4f}")
print(f"R2 Score (TRAIN): {r2_train:.4f} (check for overfitting)")
print(f"Mean Absolute Error (MAE): {mae:.2f} km/h")
print(f"Median Absolute Error (MedianAE): {medianae:.2f} km/h")
print(f"RMSE: {rmse:.2f} km/h")
print(f"Max Error: {max_er:.2f} km/h")
for t in ERROR_THRESHOLDS:
    print(f"Errors > {t} km/h: {error_pcts[t]:.2f}%")


# Calculate R2 per iteration
train_metric_key = "mphe" if USE_HUBER_LOSS else "rmse"
test_metric_key = "mphe" if USE_HUBER_LOSS else "rmse"
train_mse = np.array(evals_result["train"][train_metric_key]) ** 2
test_mse = np.array(evals_result["test"][test_metric_key]) ** 2
var_y = np.var(y_test)
train_r2_history = 1 - (train_mse / np.var(y_train))
test_r2_history = 1 - (test_mse / var_y)


# Visualizations
if "1" in show_graphs:
    plt.figure(figsize=PLOT_FIGSIZE_MAIN)

    # Plot 1: Train & Test Loss (RMSE)
    plt.subplot(2, 2, 1)
    metric_name = "rmse"
    epochs = len(evals_result["train"][metric_name])
    plt.plot(
        range(epochs),
        evals_result["train"][metric_name],
        label="Train Loss",
        color="blue",
    )
    plt.plot(
        range(epochs),
        evals_result["test"][metric_name],
        label="Test Loss",
        color="red",
        linestyle="--",
    )
    plt.title("Loss Evolution (RMSE)")
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot 2: Train & Test R2
    plt.subplot(2, 2, 2)
    plt.plot(range(epochs), train_r2_history, label="Train R2", color="green")
    plt.plot(
        range(epochs),
        test_r2_history,
        label="Test R2",
        color="darkorange",
        linestyle="--",
    )

    # Zoom on the converged region - use last PLOT_R2_FOCUS_LAST_PCT of iterations
    last_half_idx = int(epochs * (1 - PLOT_R2_FOCUS_LAST_PCT))
    train_vals = train_r2_history[last_half_idx:]
    test_vals = test_r2_history[last_half_idx:]

    all_vals = np.concatenate([train_vals, test_vals])
    r2_min = np.min(all_vals)
    r2_max = np.max(all_vals)

    # Add padding below and above
    y_min = r2_min - (r2_max - r2_min) * PLOT_R2_Y_PADDING_LOW
    y_max = r2_max + (r2_max - r2_min) * PLOT_R2_Y_PADDING_HIGH
    plt.ylim(y_min, min(PLOT_R2_MAX_Y_CAP, y_max))

    plt.title("Convergence Focus (R2 Score)")
    plt.xlabel("Number of Trees")
    plt.ylabel("R2 (Zoomed)")
    plt.legend()
    plt.grid(alpha=0.3, which="both")

    # Plot 3: Error Distribution
    plt.subplot(2, 2, 3)
    residuals = y_test - y_pred
    sns.histplot(residuals, bins=PLOT_ERROR_HIST_BINS, kde=True, color="purple")
    plt.axvline(x=0, color="black", linestyle="--")
    plt.title("Error Distribution (Residuals)")

    # Plot 4: Feature Importance
    plt.subplot(2, 2, 4)
    importance_dict = model.get_score(importance_type="weight")
    importances = [importance_dict.get(f, 0) for f in features]
    sns.barplot(
        x=importances, y=features, hue=features, palette="viridis", legend=False
    )  # type: ignore[arg-type]
    plt.title("Feature Importance")

    plt.tight_layout()
    plt.show()

if "2" in show_graphs:
    # Final Plot: Temporal Zoom
    plt.figure(figsize=PLOT_FIGSIZE_TEMPORAL)
    plt.plot(
        y_test.values[:PLOT_TEMPORAL_ZOOM_ROWS],
        label="Actual",
        color="black",
        alpha=0.7,
    )
    plt.plot(
        y_pred[:PLOT_TEMPORAL_ZOOM_ROWS],
        label="Predicted",
        color="orange",
        linestyle="--",
    )
    plt.title("Zoom 50h: Actual vs Predicted")
    plt.legend()
    plt.show()


print("--- Naivity Score ---")
first_lag_col = LAG_PREFIX.format(SHIFT_VALUES[0] * 10)
y_pred_naiv = test_df.loc[y_test.index, first_lag_col].values

print(f"Naive Model R2 Score: {r2_score(y_test, y_pred_naiv):.4f}")
print(
    f"Naive Model Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_naiv):.2f} km/h"
)


os.makedirs("models", exist_ok=True)
model.save_model(f"models/{str(os.path.basename(__file__))[:-3]}.ubj")
