from __future__ import annotations
import argparse, pathlib, polars as pl
from lib.common.settings import load_settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="NBA")
    ap.add_argument("--decision_offset_min", type=int, default=None)
    args = ap.parse_args()
    s = load_settings()

    wh = pathlib.Path(s.paths["warehouse"]) / args.league
    wh.mkdir(parents=True, exist_ok=True)

    features_path = wh / "features.parquet"
    results_path = wh / "results.parquet"

    if not features_path.exists() or not results_path.exists():
        raise FileNotFoundError("Missing features or results parquet files")

    # --- Load data ---
    feats = pl.read_parquet(features_path)
    results = pl.read_parquet(results_path)

    offset_min = args.decision_offset_min or s.decisions["pregame_offset_min"]
    print(f"[build_labels] using decision offset {offset_min} min")

    # --- Join features with nearest snapshot before start_time_utc - offset ---
    # Compute decision_ts for each game
    games = (
        results.select(["game_id"])
        .join(feats.select(["game_id"]).unique(), on="game_id", how="inner")
        .unique()
    )

    snapshots = []
    for g in games["game_id"]:
        f = feats.filter(pl.col("game_id") == g)
        start_time = f["mins_to_start"].min()  # since mins_to_start decreases as we approach start
        # decision point is start_time - offset, but we just find closest <= offset
        f_target = (
            f.filter(pl.col("mins_to_start") >= offset_min)
            .sort("mins_to_start", descending=False)
        )
        if f_target.height == 0:
            # fallback to last quote before start
            f_target = f.sort("mins_to_start").tail(1)
            source = "fallback"
        else:
            f_target = f_target.head(1)
            source = "primary"

        f_target = f_target.with_columns(pl.lit(source).alias("source"))
        snapshots.append(f_target)

    if not snapshots:
        raise RuntimeError("No decision snapshots found")
    labels = pl.concat(snapshots, how="vertical_relaxed")

    # --- Attach winner label ---
    labels = labels.join(
        results.select(["game_id", "winner"]),
        on="game_id",
        how="left",
    ).with_columns([
        (pl.col("runner") == pl.col("winner")).cast(pl.Int8).alias("y"),
        pl.col("ts_utc").alias("decision_ts")
    ])

    # --- Clean and order ---
    keep_cols = [
        "decision_ts", "game_id", "runner",
        "imp_prob_mean", "imp_prob_vigadj",
        "vig_spread", "home_away_ratio", "mins_to_start",
        "y", "source"
    ]
    labels = labels.select([c for c in keep_cols if c in labels.columns])
    labels = labels.sort(["game_id", "runner"])

    out_path = wh / "labels.parquet"
    out_path.unlink(missing_ok=True)
    labels.write_parquet(out_path)

    print(f"[build_labels] wrote {out_path} rows={labels.height}")
    print(labels)

if __name__ == "__main__":
    main()
