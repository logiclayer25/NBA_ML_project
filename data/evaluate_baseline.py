import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("nba_ml_games.csv")

df["game_date"] = pd.to_datetime(df["game_date"])

# Same test period as our ML model
test = df[df["game_date"] >= "2018-01-01"].copy()

# Sportsbook prediction
baseline_pred = test["total_line"]

# Actual result
actual = test["actual_total"]

# Metrics
mae = mean_absolute_error(actual, baseline_pred)
rmse = np.sqrt(mean_squared_error(actual, baseline_pred))
r2 = r2_score(actual, baseline_pred)

print("SPORTSBOOK BASELINE")
print("--------------------")
print(f"Test games: {len(test)}")
print(f"MAE:  {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.4f}")
