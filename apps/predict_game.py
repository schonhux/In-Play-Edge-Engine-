from __future__ import annotations
import argparse, joblib, pathlib
import numpy as np, polars as pl
from lib.common.settings import load_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="NBA")
    ap.add_argument("--home_team", required=True)
    ap.add_argument("--away_team", required=True)
    ap.add_argument("--home_price", type=float, required=True, help="Decimal odds for home team")
    ap.add_argument("--away_price", type=float, required=True, help="Decimal odds for away team")
    args = ap.parse_args()

    s = load_settings()
    league = args.league
    art = pathlib.Path(s.paths["artifacts"]) / league

    # --- Load model + calibrator ---
    model = joblib.load(art / "model.joblib")
    calibrator = joblib.load(art / "calibrator.joblib")

    # --- Create a synthetic feature row (pregame) ---
    # For now we assume market-level features from input odds
    home_imp = 1 / args.home_price
    away_imp = 1 / args.away_price
    vig_spread = (home_imp + away_imp) - 1
    home_away_ratio = home_imp / away_imp - 1
    mins_to_start = 30  # static for demo; later you can pull live game start times

    features = np.array([[home_imp, home_imp / (home_imp + away_imp),
                          vig_spread, home_away_ratio, mins_to_start]])

    # --- Predict win probability for HOME ---
    p_raw = model.predict(features[:, :3])  # adjust slice to match training features
    p_hat = calibrator.predict(p_raw)

    # --- Derive fair prices + EVs ---
    fair_price_home = 1 / p_hat[0]
    fair_price_away = 1 / (1 - p_hat[0])

    EV_home = p_hat[0] * (args.home_price - 1) - (1 - p_hat[0])
    EV_away = (1 - p_hat[0]) * (args.away_price - 1) - p_hat[0]

    print("\n==============================")
    print(f"🏀  {args.away_team} @ {args.home_team}")
    print("==============================")
    print(f"Model Win Prob (HOME): {p_hat[0]*100:.2f}%")
    print(f"Fair Price (HOME):     {fair_price_home:.3f}")
    print(f"Fair Price (AWAY):     {fair_price_away:.3f}")
    print(f"Book Price (HOME):     {args.home_price:.3f}")
    print(f"Book Price (AWAY):     {args.away_price:.3f}")
    print(f"EV (HOME):             {EV_home*100:.2f}%")
    print(f"EV (AWAY):             {EV_away*100:.2f}%")

    if EV_home > EV_away and EV_home > 0:
        print(f"✅ Recommended Bet: HOME ({args.home_team}) — {EV_home*100:.2f}% edge")
    elif EV_away > 0:
        print(f"✅ Recommended Bet: AWAY ({args.away_team}) — {EV_away*100:.2f}% edge")
    else:
        print("⚠️  No +EV opportunity found.")
    print("==============================\n")


if __name__ == "__main__":
    main()
