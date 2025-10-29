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

    csv_files = list(vendors.glob("odds*.csv"))
    if not csv_files:
        print(f"[nba_odds] WARN: no odds*.csv files found in {vendors}")
        return

    dfs = []
    for f in csv_files:
        # Force ts_utc as string; parse later to avoid tz confusion
        df = pl.read_csv(f, try_parse_dates=False, schema_overrides={"ts_utc": pl.Utf8})

        # Convert American odds if needed
        if "price_decimal" not in df.columns and "price_american" in df.columns:
            df = df.with_columns(
                pl.when(pl.col("price_american") >= 0)
                .then(1 + (pl.col("price_american") / 100.0))
                .otherwise(1 + (100.0 / (-pl.col("price_american"))))
                .alias("price_decimal")
                .cast(pl.Float64)
            )

        # If market column missing, create and fill with "moneyline"
        if "market" not in df.columns:
            df = df.with_columns(pl.lit("moneyline").alias("market"))

        dfs.append(df)

    # Combine all CSVs
    df = pl.concat(dfs, how="vertical_relaxed")

    # --- Normalize types ---
    df = df.with_columns([
        # Parse as UTC, drop timezone for naive datetime
        pl.col("ts_utc")
            .str.strptime(pl.Datetime(time_zone="UTC"), strict=False)
            .dt.replace_time_zone(None)
            .alias("ts_utc"),
        pl.col("game_id").cast(pl.Utf8),
        pl.col("book").cast(pl.Utf8).str.to_lowercase(),
        pl.col("market").cast(pl.Utf8),
        pl.col("runner").cast(pl.Utf8).str.to_uppercase(),
        pl.col("price_decimal").cast(pl.Float64),
    ])

    # Filter and clean
    df = (
        df.filter(
            (pl.col("market") == "moneyline")
            & (pl.col("runner").is_in(["HOME", "AWAY"]))
        )
        .select(["ts_utc", "game_id", "book", "market", "runner", "price_decimal"])
        .unique(subset=["ts_utc", "game_id", "book", "runner"])
        .sort(["game_id", "ts_utc", "book", "runner"])
    )

    # Save to warehouse
    out_path = wh / "ticks.parquet"
    out_path.unlink(missing_ok=True)
    df.write_parquet(out_path)
    print(f"[nba_odds] wrote {out_path} rows={df.height}")

if __name__ == "__main__":
    main()
