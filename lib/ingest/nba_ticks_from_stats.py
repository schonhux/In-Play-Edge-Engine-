from __future__ import annotations
import pathlib
import polars as pl
import numpy as np
from lib.common.settings import load_settings

def main():
    s = load_settings()
    league = "NBA"
    wh = pathlib.Path(s.paths["warehouse"]) / league
    stats_path = wh / "team_stats.parquet"
    out_path = wh / "ticks.parquet"

    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} not found. Run nba_stats.py first.")

    print(f"[nba_ticks_from_stats] 🚀 Generating synthetic odds from {stats_path.name}")

    df = pl.read_parquet(stats_path)

    # Parse the date column safely
    if df["date"].dtype != pl.Datetime:
        df = df.with_columns([
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Datetime, "%Y-%m-%d", strict=False)
        ])

    # Simulate implied home win probability using historical margin
    df = df.with_columns([
        (1 / (1 + np.exp(-pl.col("margin") / 10))).alias("p_home")
    ])

    # Add synthetic odds records for both HOME and AWAY
    ticks_home = df.select([
        pl.col("date").alias("ts_utc"),
        pl.col("season").cast(pl.Utf8).alias("game_id"),
        pl.lit("pinnacle").alias("book"),
        pl.lit("moneyline").alias("market"),
        pl.lit("HOME").alias("runner"),
        (1.0 / pl.col("p_home")).alias("price_decimal"),
        pl.col("home_team_id"),
        pl.col("away_team_id"),
    ])

    ticks_away = df.select([
        pl.col("date").alias("ts_utc"),
        pl.col("season").cast(pl.Utf8).alias("game_id"),
        pl.lit("pinnacle").alias("book"),
        pl.lit("moneyline").alias("market"),
        pl.lit("AWAY").alias("runner"),
        (1.0 / (1 - pl.col("p_home"))).alias("price_decimal"),
        pl.col("home_team_id"),
        pl.col("away_team_id"),
    ])

    ticks = pl.concat([ticks_home, ticks_away], how="vertical_relaxed")
    ticks = ticks.sort(["game_id", "ts_utc", "runner"])

    # Save output
    out_path.unlink(missing_ok=True)
    ticks.write_parquet(out_path)

    print(f"[nba_ticks_from_stats] ✅ wrote synthetic ticks → {out_path} rows={ticks.height}")
    print(ticks.head(10))

if __name__ == "__main__":
    main()
