import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("nba_ml_games.csv")

df["game_date"] = pd.to_datetime(df["game_date"])


# Features
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
    "away_blk_avg"
]


# Predict difference from sportsbook line
df["line_error"] = df["actual_total"] - df["total_line"]


# Chronological split
train = df[df["game_date"] < "2016-01-01"]

validation = df[
    (df["game_date"] >= "2016-01-01") &
    (df["game_date"] < "2018-01-01")
]

test = df[df["game_date"] >= "2018-01-01"]


X_train = train[features]
y_train = train["line_error"]

X_validation = validation[features]
y_validation = validation["line_error"]

X_test = test[features]
y_test = test["line_error"]


print("Training games:", len(train))
print("Validation games:", len(validation))
print("Test games:", len(test))


# Model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# Train
model.fit(X_train, y_train)


# Predict line error
validation_error_pred = model.predict(X_validation)
test_error_pred = model.predict(X_test)


# Convert back to total-point prediction
validation_pred = (
    validation["total_line"].values +
    validation_error_pred
)

test_pred = (
    test["total_line"].values +
    test_error_pred
)


# Evaluation
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


# Feature importance
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