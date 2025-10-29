from __future__ import annotations
import sys, os, pathlib
import polars as pl
from lib.common.settings import load_settings


def main():
    print("[nba_stats] 🚀 Merging historical + modern NBA data (2005–2025)")
    s = load_settings()
    league = "NBA"

    vendors = pathlib.Path(s.paths["vendors"]) / league / "raw"
    wh = pathlib.Path(s.paths["warehouse"]) / league
    wh.mkdir(parents=True, exist_ok=True)

    legacy = vendors / "nba_games.csv"
    modern = vendors / "nba_games_modern.csv"

    dfs = []

    # --- Load legacy data (2005–2020) ---
    if legacy.exists():
        print(f"[nba_stats] Loading legacy file: {legacy.name}")
        try:
            df_old = pl.read_csv(
                legacy,
                infer_schema_length=20000,
                ignore_errors=True
            ).select([
                pl.col("GAME_DATE_EST").alias("date"),
                pl.col("SEASON").alias("season"),
                pl.col("TEAM_ID_home").alias("home_team_id"),
                pl.col("TEAM_ID_away").alias("away_team_id"),
                pl.col("PTS_home").cast(pl.Float64),
                pl.col("PTS_away").cast(pl.Float64),
                pl.col("FG_PCT_home").cast(pl.Float64),
                pl.col("FG_PCT_away").cast(pl.Float64),
                pl.col("HOME_TEAM_WINS").cast(pl.Int8).alias("home_win")
            ])
            dfs.append(df_old)
        except Exception as e:
            print(f"[nba_stats] ⚠️ Error loading legacy file: {e}")

    # --- Load modern data (2021–2025) ---
    if modern.exists():
        print(f"[nba_stats] Loading modern file: {modern.name}")
        try:
            df_new = pl.read_csv(
                modern,
                infer_schema_length=20000,
                ignore_errors=True
            )

            # Map home/away teams per matchup
            # Split by "vs." or "@" to infer home/away
            df_home = df_new.filter(pl.col("MATCHUP").str.contains("vs.")).rename({
                "TEAM_ID": "home_team_id",
                "PTS": "PTS_home",
                "FG_PCT": "FG_PCT_home",
                "team_win": "home_win"
            })
            df_away = df_new.filter(pl.col("MATCHUP").str.contains("@")).rename({
                "TEAM_ID": "away_team_id",
                "PTS": "PTS_away",
                "FG_PCT": "FG_PCT_away"
            })

            # Join home + away by GAME_ID
            df_joined = df_home.join(
                df_away,
                on="GAME_ID",
                how="inner",
                suffix="_away"
            ).select([
                pl.col("GAME_DATE").alias("date"),
                (pl.col("SEASON_ID").cast(pl.Utf8)).alias("season"),
                "home_team_id",
                "away_team_id",
                "PTS_home",
                "PTS_away",
                "FG_PCT_home",
                "FG_PCT_away",
                "home_win"
            ])
            dfs.append(df_joined)
        except Exception as e:
            print(f"[nba_stats] ⚠️ Error loading modern file: {e}")

    if not dfs:
        print("[nba_stats] ❌ No NBA data found. Exiting.")
        sys.exit(1)

    df = pl.concat(dfs, how="vertical_relaxed")

    # --- Derived features ---
    df = df.with_columns([
        (pl.col("PTS_home") - pl.col("PTS_away")).alias("margin"),
        (pl.col("FG_PCT_home") - pl.col("FG_PCT_away")).alias("fg_diff")
    ])

    out_path = wh / "team_stats.parquet"
    out_path.unlink(missing_ok=True)
    df.write_parquet(out_path)
    print(f"[nba_stats] ✅ wrote {out_path} rows={df.height}")
    print("[nba_stats] 🏁 Done.")


if __name__ == "__main__":
    main()
