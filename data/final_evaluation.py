import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("nba_ml_games.csv")
df["game_date"] = pd.to_datetime(df["game_date"])

# ==============================
# FEATURE ENGINEERING
# ==============================

df["pts_diff"] = df["home_pts_avg"] - df["away_pts_avg"]
df["reb_diff"] = df["home_reb_avg"] - df["away_reb_avg"]
df["ast_diff"] = df["home_ast_avg"] - df["away_ast_avg"]
df["fg_pct_diff"] = df["home_fg_pct_avg"] - df["away_fg_pct_avg"]
df["fg3_pct_diff"] = df["home_fg3_pct_avg"] - df["away_fg3_pct_avg"]
df["ft_pct_diff"] = df["home_ft_pct_avg"] - df["away_ft_pct_avg"]
df["tov_diff"] = df["home_tov_avg"] - df["away_tov_avg"]
df["stl_diff"] = df["home_stl_avg"] - df["away_stl_avg"]
df["blk_diff"] = df["home_blk_avg"] - df["away_blk_avg"]

df["combined_pts_avg"] = df["home_pts_avg"] + df["away_pts_avg"]
df["combined_reb_avg"] = df["home_reb_avg"] + df["away_reb_avg"]
df["combined_ast_avg"] = df["home_ast_avg"] + df["away_ast_avg"]
df["combined_tov_avg"] = df["home_tov_avg"] + df["away_tov_avg"]


# ==============================
# FEATURES
# ==============================

features = [
    "total_line",

    "home_pts_avg",
    "away_pts_avg",
    "home_reb_avg",
    "away_reb_avg",
    "home_ast_avg",
    "away_ast_avg",
    "home_fg_pct_avg",
    "away_fg_pct_avg",
    "home_fg3_pct_avg",
    "away_fg3_pct_avg",
    "home_ft_pct_avg",
    "away_ft_pct_avg",
    "home_tov_avg",
    "away_tov_avg",
    "home_stl_avg",
    "away_stl_avg",
    "home_blk_avg",
    "away_blk_avg",

    "pts_diff",
    "reb_diff",
    "ast_diff",
    "fg_pct_diff",
    "fg3_pct_diff",
    "ft_pct_diff",
    "tov_diff",
    "stl_diff",
    "blk_diff",

    "combined_pts_avg",
    "combined_reb_avg",
    "combined_ast_avg",
    "combined_tov_avg"
]

df["line_error"] = df["actual_total"] - df["total_line"]


# ==============================
# CHRONOLOGICAL SPLIT
# ==============================

train = df[df["game_date"] < "2016-01-01"]

validation = df[
    (df["game_date"] >= "2016-01-01") &
    (df["game_date"] < "2018-01-01")
]

test = df[df["game_date"] >= "2018-01-01"]


print("==============================")
print("FINAL MODEL EVALUATION")
print("==============================")

print("Training games:", len(train))
print("Validation games:", len(validation))
print("Test games:", len(test))


# ==============================
# TRAIN MODEL
# ==============================

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

model.fit(
    train[features],
    train["line_error"]
)


# ==============================
# TEST PREDICTION
# ==============================

predicted_error = model.predict(test[features])

predicted_total = test["total_line"].values + predicted_error
actual_total = test["actual_total"].values
sportsbook_line = test["total_line"].values


# ==============================
# METRICS
# ==============================

ml_mae = mean_absolute_error(actual_total, predicted_total)
ml_rmse = np.sqrt(mean_squared_error(actual_total, predicted_total))
ml_r2 = r2_score(actual_total, predicted_total)

baseline_mae = mean_absolute_error(actual_total, sportsbook_line)
baseline_rmse = np.sqrt(mean_squared_error(actual_total, sportsbook_line))
baseline_r2 = r2_score(actual_total, sportsbook_line)


print()
print("SPORTSBOOK BASELINE")
print("------------------------------")
print("MAE :", round(baseline_mae, 4))
print("RMSE:", round(baseline_rmse, 4))
print("R²  :", round(baseline_r2, 4))

print()
print("IMPROVED RANDOM FOREST")
print("------------------------------")
print("MAE :", round(ml_mae, 4))
print("RMSE:", round(ml_rmse, 4))
print("R²  :", round(ml_r2, 4))


# ==============================
# IMPROVEMENT
# ==============================

mae_improvement = (
    (baseline_mae - ml_mae) / baseline_mae
) * 100

print()
print("MAE improvement over sportsbook:",
      round(mae_improvement, 2), "%")


# ==============================
# SAVE PREDICTIONS
# ==============================

results = test[
    ["game_id", "game_date", "total_line", "actual_total"]
].copy()

results["predicted_total"] = predicted_total
results["model_error"] = predicted_total - results["actual_total"]
results["baseline_error"] = (
    results["total_line"] - results["actual_total"]
)

results["model_absolute_error"] = abs(
    results["model_error"]
)

results["baseline_absolute_error"] = abs(
    results["baseline_error"]
)

results.to_csv(
    "../results/test_predictions.csv",
    index=False
)


# ==============================
# SAVE METRICS
# ==============================

metrics = pd.DataFrame({
    "Model": [
        "Sportsbook Baseline",
        "Improved Random Forest"
    ],
    "MAE": [
        baseline_mae,
        ml_mae
    ],
    "RMSE": [
        baseline_rmse,
        ml_rmse
    ],
    "R2": [
        baseline_r2,
        ml_r2
    ]
})

metrics.to_csv(
    "../results/model_metrics.csv",
    index=False
)


# ==============================
# FEATURE IMPORTANCE
# ==============================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance.to_csv(
    "../results/feature_importance.csv",
    index=False
)


# ==============================
# GRAPH 1: MODEL COMPARISON
# ==============================

plt.figure(figsize=(8, 5))

models = [
    "Sportsbook",
    "Random Forest"
]

mae_values = [
    baseline_mae,
    ml_mae
]

plt.bar(models, mae_values)

plt.ylabel("MAE")
plt.title("MAE Comparison")

plt.tight_layout()

plt.savefig(
    "../results/mae_comparison.png",
    dpi=300
)

plt.close()


# ==============================
# GRAPH 2: ACTUAL VS PREDICTED
# ==============================

plt.figure(figsize=(8, 6))

plt.scatter(
    actual_total,
    predicted_total,
    alpha=0.5
)

minimum = min(
    actual_total.min(),
    predicted_total.min()
)

maximum = max(
    actual_total.max(),
    predicted_total.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Total")
plt.ylabel("Predicted Total")

plt.title("Actual vs Predicted Total Points")

plt.tight_layout()

plt.savefig(
    "../results/actual_vs_predicted.png",
    dpi=300
)

plt.close()


# ==============================
# GRAPH 3: ERROR DISTRIBUTION
# ==============================

errors = predicted_total - actual_total

plt.figure(figsize=(8, 5))

plt.hist(
    errors,
    bins=30
)

plt.xlabel("Prediction Error")
plt.ylabel("Number of Games")

plt.title("Prediction Error Distribution")

plt.tight_layout()

plt.savefig(
    "../results/error_distribution.png",
    dpi=300
)

plt.close()


# ==============================
# GRAPH 4: FEATURE IMPORTANCE
# ==============================

top_features = importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)

plt.xlabel("Importance")

plt.title("Top 15 Feature Importances")

plt.tight_layout()

plt.savefig(
    "../results/feature_importance.png",
    dpi=300
)

plt.close()


# ==============================
# FINAL OUTPUT
# ==============================

print()
print("==============================")
print("FILES CREATED")
print("==============================")

print("../results/test_predictions.csv")
print("../results/model_metrics.csv")
print("../results/feature_importance.csv")
print("../results/mae_comparison.png")
print("../results/actual_vs_predicted.png")
print("../results/error_distribution.png")
print("../results/feature_importance.png")

print()
print("Evaluation completed successfully.")