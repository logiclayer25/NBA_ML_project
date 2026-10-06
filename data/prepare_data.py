import pandas as pd

df = pd.read_csv("nba_ml_dataset.csv")

df["game_date"] = pd.to_datetime(df["game_date"])

# Sort chronologically
df = df.sort_values(["team_id", "game_date"])

stats = [
    "pts",
    "reb",
    "ast",
    "fg_pct",
    "fg3_pct",
    "ft_pct",
    "tov",
    "stl",
    "blk"
]

# Create rolling averages using ONLY previous games
for stat in stats:
    df[f"prev_{stat}_avg"] = (
        df.groupby("team_id")[stat]
        .transform(lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    )

# Keep only regular season games
df = df[df["season_type"] == "Regular Season"].copy()

# Separate home and away rows
home = df[df["is_home"] == "t"].copy()
away = df[df["is_home"] == "f"].copy()

# Rename home features
home_features = [
    "game_id",
    "game_date",
    "team_id",
    "a_team_id",
    "total_line",
    "pts"
]

home_stats = [f"prev_{stat}_avg" for stat in stats]

home = home[home_features + home_stats]

home = home.rename(columns={
    "team_id": "home_team_id",
    "a_team_id": "away_team_id",
    "pts": "home_pts"
})

home = home.rename(columns={
    f"prev_{stat}_avg": f"home_{stat}_avg"
    for stat in stats
})

# Rename away features
away_stats = ["game_id", "pts"] + [f"prev_{stat}_avg" for stat in stats]

away = away[away_stats]

away = away.rename(columns={
    "pts": "away_pts"
})

away = away.rename(columns={
    f"prev_{stat}_avg": f"away_{stat}_avg"
    for stat in stats
})

# Merge home and away rows
games = home.merge(
    away,
    on="game_id",
    how="inner"
)

# Actual total score
games["actual_total"] = games["home_pts"] + games["away_pts"]

# Remove rows where we don't have enough historical games
feature_columns = (
    [f"home_{stat}_avg" for stat in stats] +
    [f"away_{stat}_avg" for stat in stats]
)

games = games.dropna(subset=feature_columns)

# Sort chronologically
games = games.sort_values("game_date")

# Save
games.to_csv("nba_ml_games.csv", index=False)

print("===================================")
print("ML DATASET CREATED")
print("===================================")

print("Shape:", games.shape)

print("\nColumns:")
print(games.columns.tolist())

print("\nDate range:")
print(games["game_date"].min(), "to", games["game_date"].max())

print("\nSample:")
print(games.head())

print("\nSaved as nba_ml_games.csv")