from __future__ import annotations
from pathlib import Path
import polars as pl
from lib.common.settings import Settings

def main():
    s = Settings.load()
    raw_path = Path(s.raw_dir) / "toy_ticks.parquet"
    df = pl.read_parquet(raw_path)

    # sort and set grouping keys
    df = df.sort(["game_id", "runner", "ts"])
    by_keys = ["game_id", "runner"]

    # odds-based features
    feats = (
        df.with_columns([
            pl.col("odds").cast(pl.Float64).alias("odds"),
            (pl.col("odds").diff().fill_null(0)).alias("odds_diff"),
            (1.0 / pl.col("odds")).alias("implied_p"),
        ])
        .with_columns([
            pl.col("odds").rolling_mean(window_size=5).over(by_keys).alias("odds_ma_5"),
            pl.col("odds").rolling_mean(window_size=10).over(by_keys).alias("odds_ma_10"),
            pl.col("odds").rolling_mean(window_size=20).over(by_keys).alias("odds_ma_20"),
            pl.col("odds_diff").rolling_std(window_size=10).over(by_keys).alias("odds_vol_10"),
        ])
    )

    keep_cols = [
        "ts","game_id","market_id","runner",
        "odds","odds_diff","implied_p",
        "odds_ma_5","odds_ma_10","odds_ma_20","odds_vol_10"
    ]
    feats = feats.select(keep_cols)

    out_path = Path(s.warehouse_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.write_parquet(out_path)
    print("Wrote features to", out_path)

if __name__ == "__main__":
    main()
