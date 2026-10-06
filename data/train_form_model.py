import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("nba_ml_dataset.csv")

df["game_date"] = pd.to_datetime(df["game_date"])

# Only regular season games
df = df[df["season_type"] == "Regular Season"].copy()

# Sort chronologically for each team
df = df.sort_values(["team_id", "game_date"])


# ============================================================
# RECENT-FORM FEATURES
# ============================================================

stats = [
    "pts",
    "reb",
    "ast",
    "fg_pct",
    "fg3_pct",
    "tov",
    "stl",
    "blk"
]

windows = [3, 5, 10]

for stat in stats:

    for window in windows:

        df[f"prev_{stat}_{window}"] = (
            df.groupby("team_id")[stat]
            .transform(
                lambda x:
                x.shift(1)
                .rolling(window, min_periods=window)
                .mean()
            )
        )


# ============================================================
# SEPARATE HOME AND AWAY
# ============================================================

home = df[df["is_home"] == "t"].copy()
away = df[df["is_home"] == "f"].copy()


# ============================================================
# HOME FEATURES
# ============================================================

home_columns = [
    "game_id",
    "total_line",
    "pts"
]

for stat in stats:
    for window in windows:
        home_columns.append(f"prev_{stat}_{window}")

home = home[home_columns]

home = home.rename(
    columns={
        "pts": "home_pts"
    }
)

home = home.rename(
    columns={
        f"prev_{stat}_{window}":
        f"home_{stat}_{window}"
        for stat in stats
        for window in windows
    }
)


# ============================================================
# AWAY FEATURES
# ============================================================

away_columns = [
    "game_id",
    "pts"
]

for stat in stats:
    for window in windows:
        away_columns.append(f"prev_{stat}_{window}")

away = away[away_columns]

away = away.rename(
    columns={
        "pts": "away_pts"
    }
)

away = away.rename(
    columns={
        f"prev_{stat}_{window}":
        f"away_{stat}_{window}"
        for stat in stats
        for window in windows
    }
)


# ============================================================
# MERGE HOME + AWAY
# ============================================================

games = home.merge(
    away,
    on="game_id",
    how="inner"
)


# ============================================================
# TARGET
# ============================================================

games["actual_total"] = (
    games["home_pts"] +
    games["away_pts"]
)

games["line_error"] = (
    games["actual_total"] -
    games["total_line"]
)


# ============================================================
# DIFFERENCE + COMBINED FEATURES
# ============================================================

for stat in stats:

    for window in windows:

        home_col = f"home_{stat}_{window}"
        away_col = f"away_{stat}_{window}"

        games[f"{stat}_diff_{window}"] = (
            games[home_col] -
            games[away_col]
        )

        games[f"combined_{stat}_{window}"] = (
            games[home_col] +
            games[away_col]
        )


# ============================================================
# FEATURES USED BY MODEL
# ============================================================

features = [
    "total_line"
]


# Add recent-form features
for stat in stats:

    for window in windows:

        features.append(
            f"home_{stat}_{window}"
        )

        features.append(
            f"away_{stat}_{window}"
        )

        features.append(
            f"{stat}_diff_{window}"
        )

        features.append(
            f"combined_{stat}_{window}"
        )


# ============================================================
# REMOVE INSUFFICIENT-HISTORY GAMES
# ============================================================

games = games.dropna(
    subset=features
)

games = games.sort_values("game_id")


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

# We use game date from the original dataset.
# Re-create it from the home dataframe.

game_dates = df[
    df["is_home"] == "t"
][["game_id", "game_date"]]

games = games.merge(
    game_dates,
    on="game_id",
    how="left"
)

games = games.sort_values("game_date")


train = games[
    games["game_date"] < "2016-01-01"
]

validation = games[
    (games["game_date"] >= "2016-01-01") &
    (games["game_date"] < "2018-01-01")
]

test = games[
    games["game_date"] >= "2018-01-01"
]


# ============================================================
# TRAINING DATA
# ============================================================

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
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=400,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)


# Train
model.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

validation_error_pred = model.predict(
    X_validation
)

test_error_pred = model.predict(
    X_test
)


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

def evaluate(
    name,
    actual,
    predicted
):

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    r2 = r2_score(
        actual,
        predicted
    )

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

print("\nTOP 20 FEATURE IMPORTANCE")
print("--------------------")

print(
    importance.head(20).to_string(
        index=False
    )
)