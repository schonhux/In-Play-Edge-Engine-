from __future__ import annotations
import argparse, pathlib, polars as pl
from lib.common.settings import load_settings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="NBA")
    args = ap.parse_args()
    s = load_settings()

    wh = pathlib.Path(s.paths["warehouse"]) / args.league
    wh.mkdir(parents=True, exist_ok=True)

    ticks_path = wh / "ticks.parquet"
    stats_path = wh / "team_stats.parquet"
    if not ticks_path.exists() or not stats_path.exists():
        raise FileNotFoundError("ticks.parquet or team_stats.parquet missing")

    print(f"[build_features] 🚀 Building features for {args.league}")

    ticks = pl.read_parquet(ticks_path)
    stats = pl.read_parquet(stats_path)

    # Compute implied probabilities
    ticks = ticks.with_columns((1.0 / pl.col("price_decimal")).alias("imp_prob"))

    # Aggregate implied probs per game and runner
    agg = (
        ticks
        .group_by(["game_id", "runner", "home_team_id", "away_team_id"])
        .agg([
            pl.mean("imp_prob").alias("imp_prob_mean"),
            pl.max("price_decimal").alias("price_best"),
        ])
    )

    # Pivot HOME/AWAY to compute vig + ratios
    home = agg.filter(pl.col("runner") == "HOME").rename({"imp_prob_mean": "home_p"})
    away = agg.filter(pl.col("runner") == "AWAY").rename({"imp_prob_mean": "away_p"})

    joined = home.join(away, on=["game_id", "home_team_id", "away_team_id"], how="inner", suffix="_away")
    joined = joined.with_columns([
        (pl.col("home_p") + pl.col("away_p") - 1.0).alias("vig_spread"),
        ((pl.col("home_p") / pl.col("away_p")) - 1.0).alias("home_away_ratio"),
    ])

    # --- Join with team stats on team IDs + season ---
    print("[build_features] Joining with team_stats for margin + fg_diff")
    df = joined.join(
        stats.select([
            "season", "home_team_id", "away_team_id", "margin", "fg_diff", "home_win"
        ]),
        on=["home_team_id", "away_team_id"],
        how="left",
    )

    # --- Add target label (1 if home win else 0) ---
    df = df.with_columns([
        (pl.col("margin") > 0).cast(pl.Int8).alias("label")
    ])

    # --- Expand back to HOME/AWAY runner rows ---
    features_home = df.select([
        "game_id", "home_team_id", "away_team_id",
        pl.lit("HOME").alias("runner"),
        pl.col("home_p").alias("imp_prob_mean"),
        pl.col("vig_spread"),
        pl.col("home_away_ratio"),
        pl.col("margin"), pl.col("fg_diff"), pl.col("label")
    ])

    features_away = df.select([
        "game_id", "home_team_id", "away_team_id",
        pl.lit("AWAY").alias("runner"),
        pl.col("away_p").alias("imp_prob_mean"),
        pl.col("vig_spread"),
        pl.col("home_away_ratio"),
        (-pl.col("margin")).alias("margin"),  # inverse perspective
        (-pl.col("fg_diff")).alias("fg_diff"),
        (1 - pl.col("label")).alias("label")
    ])

    features = pl.concat([features_home, features_away], how="vertical_relaxed")

    features = features.filter(
        (pl.col("imp_prob_mean") > 0)
        & (pl.col("imp_prob_mean") <= 1)
        & (pl.col("vig_spread") < 0.3)
        & (pl.col("margin").is_not_null())
    )

    out_path = wh / "features.parquet"
    out_path.unlink(missing_ok=True)
    features.write_parquet(out_path)

    print(f"[build_features] ✅ wrote {out_path} rows={features.height}")
    print(features.head(10))

if __name__ == "__main__":
    main()
