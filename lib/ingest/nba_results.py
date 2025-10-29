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

    src = vendors / "results.csv"
    if not src.exists():
        print(f"[nba_results] WARN: {src} not found. Expected columns: "
              "game_id, final_home_score, final_away_score, (optional) winner")
        return

    df = pl.read_csv(src)
    if "winner" not in df.columns:
        df = df.with_columns([
            pl.when(pl.col("final_home_score") > pl.col("final_away_score")).then("HOME").otherwise("AWAY").alias("winner")
        ])
    df = df.select(["game_id","final_home_score","final_away_score","winner"]).with_columns([
        pl.col("game_id").cast(pl.Utf8), pl.col("winner").cast(pl.Utf8)
    ])
    df.write_parquet(wh / "results.parquet")
    print(f"[nba_results] wrote {wh/'results.parquet'} rows={df.height}")

if __name__ == "__main__":
    main()
