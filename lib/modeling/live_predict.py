import os
import polars as pl
import joblib
import numpy as np
import lightgbm as lgb
from lib.modeling.utils import prob_to_moneyline  # same helper as before


def live_predict(team1: str, team2: str):
    stats = pl.read_parquet("data/warehouse/NBA/current_team_stats.parquet")
    odds = pl.read_parquet("data/warehouse/NBA/live_odds.parquet")

    # merge team stats
    t1 = stats.filter(pl.col("TEAM_NAME") == team1).to_dicts()[0]
    t2 = stats.filter(pl.col("TEAM_NAME") == team2).to_dicts()[0]

    fg_diff = t1["FG_PCT"] - t2["FG_PCT"]
    margin = t1["PLUS_MINUS"] - t2["PLUS_MINUS"]
    home_away_ratio = 1.0  # simple flag for home advantage

    X = np.array([[home_away_ratio, fg_diff, margin]])

    # --- Load model + calibrator ---
    model_path = "artifacts/NBA/model.joblib"
    cal_path = "artifacts/NBA/calibrator.joblib"

    if not os.path.exists(model_path):
        raise FileNotFoundError("Model not found at artifacts/NBA/model.joblib — retrain first with make train LEAGUE=NBA")
    if not os.path.exists(cal_path):
        raise FileNotFoundError("Calibrator not found at artifacts/NBA/calibrator.joblib — retrain first with make train LEAGUE=NBA")

    model = joblib.load(model_path)
    cal = joblib.load(cal_path)

    odds = odds.with_columns([
    pl.col("home_team").str.replace("LA Lakers", "Los Angeles Lakers"),
    pl.col("away_team").str.replace("LA Lakers", "Los Angeles Lakers"),
])

    # --- Predict win probability ---
    model_prob = float(cal.predict_proba(X)[:, 1][0])


    # --- Get live sportsbook odds ---
    row = odds.filter(
        (pl.col("home_team") == team1) & (pl.col("away_team") == team2)
    ).sort("book").head(1)

    if row.height == 0:
        print(f"No odds yet for {team1} vs {team2}")
        return

    home_odds = float(row["home_odds"][0])
    implied = 1 / (1 + (home_odds if home_odds > 0 else abs(100 / home_odds)))
    fair_moneyline = prob_to_moneyline(model_prob)
    edge = (model_prob - implied) * 100

    # --- Output results ---
    print(f"\n🏀 {team1} vs {team2}")
    print(f"DraftKings Odds: {home_odds}")
    print(f"Implied Prob: {implied:.3f}")
    print(f"Model Prob: {model_prob:.3f}")
    print(f"Fair Moneyline: {fair_moneyline:.0f}")
    print(f"Edge: {edge:+.2f}%\n")


if __name__ == "__main__":
    live_predict("Golden State Warriors", "Los Angeles Clippers")
