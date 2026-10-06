import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Load ML dataset
df = pd.read_csv("nba_ml_games.csv")

# Convert date
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

target = "actual_total"

# Chronological split
train = df[df["game_date"] < "2016-01-01"]
validation = df[
    (df["game_date"] >= "2016-01-01") &
    (df["game_date"] < "2018-01-01")
]
test = df[df["game_date"] >= "2018-01-01"]

X_train = train[features]
y_train = train[target]

X_validation = validation[features]
y_validation = validation[target]

X_test = test[features]
y_test = test[target]

print("Training games:", len(train))
print("Validation games:", len(validation))
print("Test games:", len(test))

# Model
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Validation predictions
validation_pred = model.predict(X_validation)

# Test predictions
test_pred = model.predict(X_test)

# Evaluation function
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)

    print(f"\n{name}")
    print("--------------------")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}")


evaluate("VALIDATION RESULTS", y_validation, validation_pred)
evaluate("TEST RESULTS", y_test, test_pred)

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
print(importance.to_string(index=False))