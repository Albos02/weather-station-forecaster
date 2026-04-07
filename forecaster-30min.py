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

import time, os
import xgboost as xgb


# Constants
TARGET_COLUMN = "vitesse_vent_moyenne_10min_kmh"
TARGET_DISTANCE = 3  # * 10 = prediction horizon in minutes
SHIFT_VALUES = [1, 2, 3, 4, 5, 6]
PRESSURE_SHIFT = 36
TRAIN_SPLIT_RATIO = 0.8
NUM_BOOST_ROUNDS = 200
MODEL_MAX_DEPTH = 5
MODEL_ETA = 0.1
RANDOM_STATE = 42

# 1. Load and sort data
df = pd.read_parquet("data/processed_windspeed.parquet", engine="pyarrow")
df["horodatage_référence"] = pd.to_datetime(df["horodatage_référence"], dayfirst=True)
df = df.sort_values("horodatage_référence").reset_index(drop=True)

# 2. Handle NaN values - linear interpolation
cols_num = df.select_dtypes(include=[np.number]).columns
df[cols_num] = df[cols_num].interpolate(method="linear").bfill()

# 3. Feature Engineering - Hour, Month and Time Lags
df["hour"] = df["horodatage_référence"].dt.hour
df["year_day_pct"] = df["horodatage_référence"].dt.dayofyear / 365

target = TARGET_COLUMN

df["target_30min"] = df[target].shift(-TARGET_DISTANCE)

df["wind_10min_ago"] = df[target].shift(1)
df["wind_20min_ago"] = df[target].shift(2)
df["wind_30min_ago"] = df[target].shift(3)
df["wind_40min_ago"] = df[target].shift(4)
df["wind_50min_ago"] = df[target].shift(5)
df["wind_60min_ago"] = df[target].shift(6)

df["wind_trend_10min"] = df[target] - df["wind_10min_ago"]
df["wind_trend_20min"] = df[target] - df["wind_20min_ago"]
df["wind_trend_30min"] = df[target] - df["wind_30min_ago"]
df["wind_trend_40min"] = df[target] - df["wind_40min_ago"]
df["wind_trend_50min"] = df[target] - df["wind_50min_ago"]
df["wind_trend_60min"] = df[target] - df["wind_60min_ago"]

df["pressure_6h_ago"] = df["pression_barométrique_qfe"].shift(PRESSURE_SHIFT)
df["pressure_trend_6h"] = df["pression_barométrique_qfe"] - df["pressure_6h_ago"]

# Drop rows with NaN in features used
features_to_check = [
    "wind_10min_ago",
    "wind_20min_ago",
    "wind_30min_ago",
    "wind_40min_ago",
    "wind_50min_ago",
    "wind_60min_ago",
    "wind_trend_10min",
    "wind_trend_20min",
    "wind_trend_30min",
    "wind_trend_40min",
    "wind_trend_50min",
    "wind_trend_60min",
    "direction_du_vent_moyenne_10min",
    "hour",
    "year_day_pct",
    "pressure_trend_6h",
    "humidité",
    "température_air",
    "rafale_3s_maximum_kmh",
]
df = df.dropna(subset=features_to_check)


# Feature selection
features = [
    "wind_10min_ago",
    "wind_20min_ago",
    "wind_30min_ago",
    "wind_40min_ago",
    "wind_50min_ago",
    "wind_60min_ago",
    "wind_trend_10min",
    "wind_trend_20min",
    "wind_trend_30min",
    "wind_trend_40min",
    "wind_trend_50min",
    "wind_trend_60min",
    "direction_du_vent_moyenne_10min",
    "hour",
    "year_day_pct",
    "pressure_trend_6h",
    "humidité",
    "température_air",
    "rafale_3s_maximum_kmh",
]

# Ensure columns exist before split
features = [f for f in features if f in df.columns]

# Chronological split (past for training)
split_idx = int(len(df) * TRAIN_SPLIT_RATIO)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]


X_train, y_train = train_df[features], train_df["target_30min"]
test_df_shifted = test_df[features].shift(10).dropna()
y_test = test_df["target_30min"].shift(10).iloc[: len(test_df_shifted)].dropna()
X_test = test_df_shifted.loc[y_test.index]

# Prepare data in DMatrix format (GPU optimized)
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)


# Model configuration for GPU
params = {
    "max_depth": MODEL_MAX_DEPTH,
    "eta": MODEL_ETA,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "random_state": RANDOM_STATE,
}

evals_result = {}

print(f"Starting GPU training on {len(X_train)} rows...")
start_time = time.time()

model = xgb.train(
    params,
    dtrain,
    num_boost_round=NUM_BOOST_ROUNDS,
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


y_test = y_test[:-3]
y_pred = y_pred[3:]


# Evaluation
mae = mean_absolute_error(y_test, y_pred)
medianae = median_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
r2_train = r2_score(y_train, y_pred_train)
max_er = max_error(y_test, y_pred)
abs_errors = np.abs(y_test - y_pred)
pct_above_2kmh = np.mean(abs_errors > 2) * 100
pct_above_5kmh = np.mean(abs_errors > 5) * 100


print(f"--- RESULTS ---")
print(f"R2 Score (TEST): {r2:.4f}")
print(f"R2 Score (TRAIN): {r2_train:.4f} (check for overfitting)")
print(f"Mean Absolute Error (MAE): {mae:.2f} km/h")
print(f"Median Absolute Error (MedianAE): {medianae:.2f} km/h")
print(f"RMSE: {rmse:.2f} km/h")
print(f"Max Error: {max_er:.2f} km/h")
print(f"Errors > 2 km/h: {pct_above_2kmh:.1f}%")
print(f"Errors > 5 km/h: {pct_above_5kmh:.1f}%")


# Calculate R2 per iteration
train_mse = np.array(evals_result["train"]["rmse"]) ** 2
test_mse = np.array(evals_result["test"]["rmse"]) ** 2
var_y = np.var(y_test)
train_r2_history = 1 - (train_mse / np.var(y_train))
test_r2_history = 1 - (test_mse / var_y)


# Visualizations
plt.figure(figsize=(18, 15))

# Plot 1: Train & Test Loss (RMSE)
plt.subplot(2, 2, 1)
metric_name = "rmse"
epochs = len(evals_result["train"][metric_name])
plt.plot(
    range(epochs), evals_result["train"][metric_name], label="Train Loss", color="blue"
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
    range(epochs), test_r2_history, label="Test R2", color="darkorange", linestyle="--"
)

# Zoom on the converged region - use last 50% of iterations
last_half_idx = epochs // 2
train_vals = train_r2_history[last_half_idx:]
test_vals = test_r2_history[last_half_idx:]

all_vals = np.concatenate([train_vals, test_vals])
r2_min = np.min(all_vals)
r2_max = np.max(all_vals)

# Add 20% padding below and 5% above
y_min = r2_min - (r2_max - r2_min) * 20
y_max = r2_max + (r2_max - r2_min) * 5
plt.ylim(y_min, min(1.001, y_max))

plt.title("Convergence Focus (R2 Score)")
plt.xlabel("Number of Trees")
plt.ylabel("R2 (Zoomed)")
plt.legend()
plt.grid(alpha=0.3, which="both")

# Plot 3: Error Distribution
plt.subplot(2, 2, 3)
residuals = y_test - y_pred
sns.histplot(residuals, bins=100, kde=True, color="purple")
plt.axvline(x=0, color="black", linestyle="--")
plt.title("Error Distribution (Residuals)")

# Plot 4: Feature Importance
plt.subplot(2, 2, 4)
importance_dict = model.get_score(importance_type="weight")
importances = [importance_dict.get(f, 0) for f in features]
sns.barplot(x=importances, y=features, hue=features, palette="viridis", legend=False)  # type: ignore[arg-type]
plt.title("Feature Importance")

plt.tight_layout()
plt.show()


# Final Plot: Temporal Zoom
plt.figure(figsize=(18, 6))
plt.plot(y_test.values[:300], label="Actual", color="black", alpha=0.7)
plt.plot(y_pred[:300], label="Predicted", color="orange", linestyle="--")
plt.title("Zoom 50h: Actual vs Predicted")
plt.legend()
plt.show()


print("--- Naivity Score ---")
y_pred_naiv = test_df.loc[y_test.index, "wind_10min_ago"].values

print(f"Naive Model R2 Score: {r2_score(y_test, y_pred_naiv):.4f}")
print(
    f"Naive Model Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred_naiv):.2f} km/h"
)


os.makedirs("models", exist_ok=True)
model.save_model(f"models/{str(os.path.basename(__file__))[:-3]}.ubj")
