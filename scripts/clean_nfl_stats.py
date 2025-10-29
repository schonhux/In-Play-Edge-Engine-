import polars as pl

def main():
    """
    Cleans raw NFL team stats into a usable format.
    Handles missing values safely and ensures numeric types.
    """
    print("[clean_nfl_stats] 🧹 Cleaning NFL team stats...")

    df = pl.read_csv("data/raw/nfl_team_stats.csv")

    df = df.rename({
        "TEAM": "TEAM_NAME",
        "OFF_YDS_PER_PLAY": "YPP",
        "DEF_YDS_PER_PLAY": "YPP_DEF",
        "POINT_DIFF": "PLUS_MINUS",
        "WIN": "WIN",
    })

    # Compute net efficiency and fill missing values
    df = df.with_columns([
        (pl.col("YPP") - pl.col("YPP_DEF")).alias("NET_YPP")
    ])

    # Drop rows with any NaN
    df = df.drop_nulls()

    # Ensure all numeric columns are floats
    df = df.with_columns([
        pl.col("YPP").cast(pl.Float64),
        pl.col("YPP_DEF").cast(pl.Float64),
        pl.col("PLUS_MINUS").cast(pl.Float64),
        pl.col("NET_YPP").cast(pl.Float64),
        pl.col("WIN").cast(pl.Int64)
    ])

    df.write_parquet("data/warehouse/NFL/current_team_stats.parquet")
    print("✅ Saved cleaned NFL stats → data/warehouse/NFL/current_team_stats.parquet")

if __name__ == "__main__":
    main()
