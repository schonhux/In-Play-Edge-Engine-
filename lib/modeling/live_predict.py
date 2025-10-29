import os
import polars as pl
import joblib
import numpy as np
from difflib import get_close_matches
from lib.modeling.utils import prob_to_moneyline
from lib.utils.team_name_map import normalize_name  # <— shared normalizer


def normalize_team_name(team_name: str, valid_names: list[str]) -> str:
    """Find the closest valid team name in the stats dataset."""
    match = get_close_matches(team_name, valid_names, n=1, cutoff=0.6)
    if not match:
        raise ValueError(f"❌ '{team_name}' not found in dataset.\nAvailable: {valid_names}")
    return match[0]


def live_predict(team1: str, team2: str):
    stats_path = "data/warehouse/NBA/current_team_stats.parquet"
    odds_path = "data/warehouse/NBA/live_odds.parquet"
    model_path = "artifacts/NBA/model.joblib"
    cal_path = "artifacts/NBA/calibrator.joblib"

    # --- Load datasets ---
    stats = pl.read_parquet(stats_path)
    odds = pl.read_parquet(odds_path)

    # Normalize team names in odds using our shared alias map
    odds = odds.with_columns([
        pl.col("home_team").map_elements(normalize_name).alias("home_team"),
        pl.col("away_team").map_elements(normalize_name).alias("away_team"),
    ])

    valid_names = stats["TEAM_NAME"].unique().to_list()
    team1 = normalize_team_name(team1, valid_names)
    team2 = normalize_team_name(team2, valid_names)

    # --- Extract features safely ---
    t1_list = stats.filter(pl.col("TEAM_NAME") == team1).to_dicts()
    t2_list = stats.filter(pl.col("TEAM_NAME") == team2).to_dicts()
    if not t1_list or not t2_list:
        raise ValueError(f"Stats not found for {team1} or {team2}")
    t1, t2 = t1_list[0], t2_list[0]

    fg_diff = t1.get("FG_PCT", 0) - t2.get("FG_PCT", 0)
    margin = t1.get("PLUS_MINUS", 0) - t2.get("PLUS_MINUS", 0)
    home_away_ratio = 1.0

    X = np.array([[home_away_ratio, fg_diff, margin]])

    # --- Load model + calibrator ---
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model at {model_path}")
    if not os.path.exists(cal_path):
        raise FileNotFoundError(f"Missing calibrator at {cal_path}")

    model = joblib.load(model_path)
    cal = joblib.load(cal_path)

    # ✅ Correct usage — the calibrator handles model internally
    model_prob = float(cal.predict_proba(X)[:, 1][0])

    # --- Find matching odds row ---
    row = odds.filter(
        (pl.col("home_team") == team1) & (pl.col("away_team") == team2)
    ).sort("book").head(1)

    if row.height == 0:
        print(f"No odds yet for {team1} vs {team2}")
        return

    home_odds = float(row["home_odds"][0])
    implied = 1 / home_odds
    fair_moneyline = prob_to_moneyline(model_prob)
    edge = (model_prob - implied) * 100

    print(f"\n🏀 {team1} vs {team2}")
    print(f"DraftKings Odds: {home_odds}")
    print(f"Implied Prob: {implied:.3f}")
    print(f"Model Prob: {model_prob:.3f}")
    print(f"Fair Moneyline: {fair_moneyline:.0f}")
    print(f"Edge: {edge:+.2f}%\n")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        team1 = sys.argv[1]
        team2 = sys.argv[2]
    else:
        print("Usage: poetry run python -m lib.modeling.live_predict 'Home Team' 'Away Team'")
        print("→ Example: poetry run python -m lib.modeling.live_predict 'Boston Celtics' 'Cleveland Cavaliers'")
        team1, team2 = "Golden State Warriors", "Los Angeles Lakers"

    live_predict(team1, team2)

