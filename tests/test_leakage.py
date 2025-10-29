import polars as pl

def test_time_sorted_and_no_future():
    df = pl.read_parquet("data/warehouse/features.parquet")
    df = df.sort(["game_id", "ts"])
    chk = df.group_by("game_id").agg(
        (pl.col("ts").cast(pl.Int64).diff().fill_null(0) >= 0).all().alias("ok")
    )
    assert chk.select(pl.col("ok").all()).item(), "Timestamps must be non-decreasing per game"
    assert all(not c.startswith("final_") for c in df.columns)
