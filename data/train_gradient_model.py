import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("nba_ml_games.csv")

df["game_date"] = pd.to_datetime(df["game_date"])


# ============================================================
# CREATE DIFFERENCE FEATURES
# ============================================================

df["pts_diff"] = (
    df["home_pts_avg"] - df["away_pts_avg"]
)

df["reb_diff"] = (
    df["home_reb_avg"] - df["away_reb_avg"]
)

df["ast_diff"] = (
    df["home_ast_avg"] - df["away_ast_avg"]
)

df["fg_pct_diff"] = (
    df["home_fg_pct_avg"] - df["away_fg_pct_avg"]
)

df["fg3_pct_diff"] = (
    df["home_fg3_pct_avg"] - df["away_fg3_pct_avg"]
)

df["ft_pct_diff"] = (
    df["home_ft_pct_avg"] - df["away_ft_pct_avg"]
)

df["tov_diff"] = (
    df["home_tov_avg"] - df["away_tov_avg"]
)

df["stl_diff"] = (
    df["home_stl_avg"] - df["away_stl_avg"]
)

df["blk_diff"] = (
    df["home_blk_avg"] - df["away_blk_avg"]
)


# ============================================================
# CREATE COMBINED FEATURES
# ============================================================

df["combined_pts_avg"] = (
    df["home_pts_avg"] + df["away_pts_avg"]
)

df["combined_reb_avg"] = (
    df["home_reb_avg"] + df["away_reb_avg"]
)

df["combined_ast_avg"] = (
    df["home_ast_avg"] + df["away_ast_avg"]
)

df["combined_tov_avg"] = (
    df["home_tov_avg"] + df["away_tov_avg"]
)


# ============================================================
# FEATURES
# ============================================================

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


# ============================================================
# TARGET: DIFFERENCE FROM SPORTSBOOK LINE
# ============================================================

df["line_error"] = (
    df["actual_total"] - df["total_line"]
)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

train = df[df["game_date"] < "2016-01-01"]

validation = df[
    (df["game_date"] >= "2016-01-01") &
    (df["game_date"] < "2018-01-01")
]

test = df[
    df["game_date"] >= "2018-01-01"
]


X_train = train[features]
y_train = train["line_error"]

X_validation = validation[features]
y_validation = validation["line_error"]

X_test = test[features]
y_test = test["line_error"]


print("Training games:", len(train))
print("Validation games:", len(validation))
print("Test games:", len(test))


# ============================================================
# GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=3,
    min_samples_leaf=10,
    random_state=42,
    loss="huber"
)


# Train
model.fit(X_train, y_train)


# ============================================================
# PREDICTIONS
# ============================================================

validation_error_pred = model.predict(X_validation)

test_error_pred = model.predict(X_test)


# Convert predicted error back to total
validation_pred = (
    validation["total_line"].values +
    validation_error_pred
)

test_pred = (
    test["total_line"].values +
    test_error_pred
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(name, actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("--------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")


evaluate(
    "VALIDATION RESULTS",
    validation["actual_total"],
    validation_pred
)

evaluate(
    "TEST RESULTS",
    test["actual_total"],
    test_pred
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)

print("\nFEATURE IMPORTANCE")
print("--------------------")

print(
    importance.to_string(index=False)
)