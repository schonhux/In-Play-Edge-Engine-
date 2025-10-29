# lib/modeling/eval.py
from __future__ import annotations
from duckdb import df
import argparse, pathlib, joblib, pandas as pd, numpy as np
from lib.common.settings import load_settings

def implied_prob_from_odds(odds: float) -> float:
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="NBA")
    ap.add_argument("--top", type=int, default=10, help="Number of matchups to display")
    ap.add_argument("--stake", type=float, default=100.0, help="Simulated bet amount per edge")
    args = ap.parse_args()

    s = load_settings()
    league = args.league
    wh = pathlib.Path(s.paths["warehouse"]) / league
    feats_path = wh / "features.parquet"
    art_dir = pathlib.Path("artifacts") / league

    if not feats_path.exists():
        raise FileNotFoundError(f"{feats_path} missing — run make features first.")
    if not (art_dir / "model.joblib").exists():
        raise FileNotFoundError(f"Trained model not found in {art_dir}")

    print(f"[eval] 🚀 Evaluating {league} model...")
    df = pd.read_parquet(feats_path)
    model = joblib.load(art_dir / "model.joblib")
    cal = joblib.load(art_dir / "calibrator.joblib")

    feature_cols = ["imp_prob_mean", "vig_spread", "home_away_ratio"]
    X = df[feature_cols].to_numpy()

    raw_probs = model.predict_proba(X)[:, 1]

# Handle calibrated model correctly — use first calibrated submodel
    try:
        calibrated_probs = cal.calibrated_classifiers_[0].calibrators_[1].predict(raw_probs.reshape(-1, 1))
    except Exception:
    # fallback if different calibration structure
        calibrated_probs = raw_probs  # use raw if calibration not compatible

    df["model_prob"] = calibrated_probs

    # Simulate moneyline odds (placeholder if not present)
    if "odds_home" not in df.columns:
        np.random.seed(42)
        df["odds_home"] = np.random.choice([110, 120, -130, -150, 100, -105], size=len(df))

    df["implied_prob"] = df["odds_home"].apply(implied_prob_from_odds)
    df["edge"] = (df["model_prob"] - df["implied_prob"]) * 100

    # Rank & filter
    ranked = (
        df.sort_values("edge", ascending=False)
          .loc[:, ["home_team_id", "away_team_id", "odds_home", "implied_prob", "model_prob", "edge"]]
          .head(args.top)
    )

    print("\n🏀 In-Play Edge Engine — Top Matchups by Model Edge")
    print("-------------------------------------------------------")
    for _, r in ranked.iterrows():
        print(
            f"{int(r.home_team_id)} vs {int(r.away_team_id)} | "
            f"Odds {r.odds_home:+} | "
            f"Implied={r.implied_prob:.3f} | "
            f"Model={r.model_prob:.3f} | "
            f"EDGE={r.edge:+.2f}%"
        )
    print("-------------------------------------------------------")

    # Optional simulated bets
    positive_edges = ranked[ranked.edge > 0]
    if not positive_edges.empty:
        total_stake = args.stake * len(positive_edges)
        print(f"\n[eval] 💰 Simulated Bets ({len(positive_edges)} games, ${args.stake:.0f} each): ${total_stake:.0f} total")
        print("Games:", ", ".join(f"{int(h)} vs {int(a)}" for h, a in zip(positive_edges.home_team_id, positive_edges.away_team_id)))
    else:
        print("\n[eval] ❌ No positive EV matchups found today.")

if __name__ == "__main__":
    main()
