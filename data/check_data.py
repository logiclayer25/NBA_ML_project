import pandas as pd
games = pd.read_csv("nba_games_all.csv")
bets = pd.read_csv("nba_betting_totals.csv")

print("\n===== GAMES DATASET =====")
print("Shape:", games.shape)
print("Unique games:", games["game_id"].nunique())
print("Date range:", games["game_date"].min(), "to", games["game_date"].max())

print("\nMissing values:")
print(games.isnull().sum())

print("\n===== BETTING DATASET =====")
print("Shape:", bets.shape)
print("Unique games:", bets["game_id"].nunique())
print("Unique books:", bets["book_name"].nunique())
print("Books:")
print(bets["book_name"].value_counts())

print("\nMissing values:")
print(bets.isnull().sum())

print("\n===== OVERLAP =====")
game_ids = set(games["game_id"].unique())
bet_game_ids = set(bets["game_id"].unique())

print("Games with betting data:", len(game_ids & bet_game_ids))
print("Games without betting data:", len(game_ids - bet_game_ids))

print("\n===== TEAM ROWS PER GAME =====")
rows_per_game = games.groupby("game_id").size()
print(rows_per_game.value_counts().sort_index())