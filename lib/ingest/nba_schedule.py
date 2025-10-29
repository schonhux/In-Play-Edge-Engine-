from __future__ import annotations
import argparse, pathlib, polars as pl
from lib.common.settings import load_settings

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="NBA")
    args = ap.parse_args()
    s = load_settings()

    vendors = pathlib.Path(s.paths["vendors"]) / args.league / "raw"
    wh = pathlib.Path(s.paths["warehouse"]) / args.league
    wh.mkdir(parents=True, exist_ok=True)

    src = vendors / "schedule.csv"
    if not src.exists():
        print(f"[nba_schedule] WARN: {src} not found. Expected columns: "
              "game_id, league, season, date_utc, start_time_utc, home_team, away_team, venue")
        return

    df = pl.read_csv(src, try_parse_dates=True)

    # Ensure types without using .str. on a temporal column
    df = df.with_columns([
        pl.col("game_id").cast(pl.Utf8),
        pl.col("start_time_utc").cast(pl.Datetime),  # safe if already datetime or string
    ])

    df.write_parquet(wh / "schedule.parquet")
    print(f"[nba_schedule] wrote {wh/'schedule.parquet'} rows={df.height}")

if __name__ == "__main__":
    main()
