from __future__ import annotations
import argparse, pathlib, json, joblib
import numpy as np, polars as pl
from lib.common.settings import load_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="NBA")
    ap.add_argument("--ev_threshold", type=float, default=None)
    ap.add_argument("--kelly_fraction", type=float, default=None)
    args = ap.parse_args()

    s = load_settings()
    league = args.league
    wh = pathlib.Path(s.paths["warehouse"]) / league
    art = pathlib.Path(s.paths["artifacts"]) / league
    rep = pathlib.Path(s.paths["reports"]) / league
    rep.mkdir(parents=True, exist_ok=True)

    # --- Load model + calibrator ---
    model = joblib.load(art / "model.joblib")
    calibrator = joblib.load(art / "calibrator.joblib")

    # --- Load features/labels ---
    df = pl.read_parquet(wh / "labels.parquet").to_pandas()
    feature_cols = ["imp_prob_mean", "imp_prob_vigadj", "vig_spread"]

    X, y = df[feature_cols].values, df["y"].values
    game_ids = df["game_id"].values
    runners = df["runner"].values

    # --- Model predictions ---
    p_raw = model.predict(X)
    p_hat = calibrator.predict(p_raw)
    df["p_hat"] = p_hat

    # --- Merge best book price ---
    ticks = pl.read_parquet(wh / "ticks.parquet").to_pandas()
    ticks_best = (
        ticks.sort_values(["game_id", "book", "runner", "ts_utc"])
        .groupby(["game_id", "runner"], as_index=False)
        .agg({"price_decimal": "max"})
        .rename(columns={"price_decimal": "P"})
    )
    df = df.merge(ticks_best, on=["game_id", "runner"], how="left")

    # --- Compute EV + fractional-Kelly stake ---
    ev_thresh = args.ev_threshold or s.betting.get("ev_threshold", 0.01)
    kelly_frac = args.kelly_fraction or s.betting.get("kelly_fraction", 0.25)
    bankroll = 1000.0
    stakes, pnl, evs = [], [], []

    for i, row in df.iterrows():
        p, P = row["p_hat"], row["P"]
        if np.isnan(P) or P <= 1.0:
            stakes.append(0.0); pnl.append(0.0); evs.append(np.nan); continue

        EV = p * (P - 1) - (1 - p)
        evs.append(EV)
        if EV < ev_thresh:
            stakes.append(0.0); pnl.append(0.0); continue

        kelly_raw = ((P - 1) * p - (1 - p)) / (P - 1)
        stake = kelly_frac * max(kelly_raw, 0) * bankroll
        outcome = 1 if row["y"] == 1 else 0
        profit = stake * (P - 1) if outcome else -stake
        bankroll += profit
        stakes.append(stake)
        pnl.append(profit)

    df["EV"], df["stake"], df["pnl"] = evs, stakes, pnl
    df["bankroll"] = np.cumsum(df["pnl"]) + 1000.0

    # --- Summary stats ---
    bet_mask = df["stake"] > 0
    ev_mean = np.nanmean(df.loc[bet_mask, "EV"]) if bet_mask.any() else 0
    n_bets = int(bet_mask.sum())
    final_bankroll = float(df["bankroll"].iloc[-1])
    roi = (final_bankroll - 1000.0) / 1000.0

    report = {
        "n_bets": n_bets,
        "avg_EV": ev_mean,
        "final_bankroll": final_bankroll,
        "ROI": roi,
        "start_bankroll": 1000.0,
    }

    # --- Write outputs ---
    signals_path = wh / "signals.parquet"
    pl.from_pandas(df).write_parquet(signals_path)
    with open(rep / "backtest.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"[backtest] wrote {signals_path}")
    print(f"[backtest] summary → {report}")


if __name__ == "__main__":
    main()
