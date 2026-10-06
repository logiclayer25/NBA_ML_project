import pandas as pd
import joblib


# ==============================
# LOAD MODEL
# ==============================

saved = joblib.load("../models/final_model.pkl")

model = saved["model"]
features = saved["features"]


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv("nba_ml_games.csv")


# ==============================
# CREATE ENGINEERED FEATURES
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


# ==============================
# SELECT GAME
# ==============================

print()
print("==============================")
print("NBA TOTAL SCORE PREDICTOR")
print("==============================")
print()

print("Dataset contains", len(df), "games.")

game_number = int(
    input(
        "Enter game number (1-{}): ".format(len(df))
    )
)

if game_number < 1 or game_number > len(df):
    print("Invalid game number.")
    exit()

game = df.iloc[game_number - 1]


# ==============================
# MODEL INPUT
# ==============================

X = pd.DataFrame(
    [[game[feature] for feature in features]],
    columns=features
)


# ==============================
# PREDICTION
# ==============================

predicted_error = model.predict(X)[0]

sportsbook_line = game["total_line"]

predicted_total = sportsbook_line + predicted_error

difference = predicted_total - sportsbook_line


# ==============================
# OVER / UNDER
# ==============================

if difference > 0:
    recommendation = "OVER"
elif difference < 0:
    recommendation = "UNDER"
else:
    recommendation = "NEUTRAL"


# ==============================
# DISPLAY RESULTS
# ==============================

print()
print("==============================")
print("PREDICTION RESULT")
print("==============================")

print("Game ID:", game["game_id"])
print("Date:", game["game_date"])

print()
print("Sportsbook Line:",
      round(sportsbook_line, 2))

print("Predicted Total:",
      round(predicted_total, 2))

print("Model Difference:",
      round(difference, 2))

print()
print("MODEL SIGNAL:", recommendation)

print()
print("Actual Total:",
      game["actual_total"])

print("==============================")