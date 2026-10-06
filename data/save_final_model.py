import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor


# Load dataset
df = pd.read_csv("nba_ml_games.csv")

df["game_date"] = pd.to_datetime(df["game_date"])


# Difference features
df["pts_diff"] = df["home_pts_avg"] - df["away_pts_avg"]
df["reb_diff"] = df["home_reb_avg"] - df["away_reb_avg"]
df["ast_diff"] = df["home_ast_avg"] - df["away_ast_avg"]
df["fg_pct_diff"] = df["home_fg_pct_avg"] - df["away_fg_pct_avg"]
df["fg3_pct_diff"] = df["home_fg3_pct_avg"] - df["away_fg3_pct_avg"]
df["ft_pct_diff"] = df["home_ft_pct_avg"] - df["away_ft_pct_avg"]
df["tov_diff"] = df["home_tov_avg"] - df["away_tov_avg"]
df["stl_diff"] = df["home_stl_avg"] - df["away_stl_avg"]
df["blk_diff"] = df["home_blk_avg"] - df["away_blk_avg"]


# Combined features
df["combined_pts_avg"] = df["home_pts_avg"] + df["away_pts_avg"]
df["combined_reb_avg"] = df["home_reb_avg"] + df["away_reb_avg"]
df["combined_ast_avg"] = df["home_ast_avg"] + df["away_ast_avg"]
df["combined_tov_avg"] = df["home_tov_avg"] + df["away_tov_avg"]


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


# Target
df["line_error"] = df["actual_total"] - df["total_line"]


# Remove missing values
df = df.dropna(subset=features)


# Train on all available historical data
X = df[features]
y = df["line_error"]


# Final model
model = RandomForestRegressor(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


model.fit(X, y)


# Save model
joblib.dump(
    {
        "model": model,
        "features": features
    },
    "../models/final_model.pkl"
)


print("Final model trained successfully.")
print("Training games:", len(df))
print("Model saved to:")
print("../models/final_model.pkl")