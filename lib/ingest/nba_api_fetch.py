from __future__ import annotations
import sys, time, pathlib
import polars as pl
from nba_api.stats.endpoints import leaguegamefinder
from lib.common.settings import load_settings


def main():
    print("[nba_api_fetch] 🚀 Fetching modern NBA game data (2021–present)")
    s = load_settings()
    league = "NBA"
    vendors = pathlib.Path(s.paths["vendors"]) / league / "raw"
    vendors.mkdir(parents=True, exist_ok=True)

    print("[nba_api_fetch] Querying API… this can take ~1–2 minutes.")
    gamefinder = leaguegamefinder.LeagueGameFinder(
        season_nullable=None,  # all seasons
        league_id_nullable="00"  # NBA
    )
    games = gamefinder.get_data_frames()[0]

    # Keep essential columns only
    df = pl.from_pandas(games)[
        [
            "GAME_DATE", "SEASON_ID", "GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION",
            "TEAM_NAME", "MATCHUP", "WL", "PTS", "FG_PCT", "FG3_PCT", "FT_PCT",
            "REB", "AST"
        ]
    ]

    # Filter seasons >= 2020–21 (ID 22020 and up)
    df = df.filter(pl.col("SEASON_ID").cast(pl.Int32) >= 22020)

    # Add helper columns
    df = df.with_columns([
        (pl.col("GAME_DATE").str.strptime(pl.Date, "%Y-%m-%d")).alias("date"),
        (pl.col("WL") == "W").cast(pl.Int8).alias("team_win")
    ])

    out_path = vendors / "nba_games_modern.csv"
    out_path.unlink(missing_ok=True)
    df.write_csv(out_path)
    print(f"[nba_api_fetch] ✅ wrote {out_path} rows={df.height}")


if __name__ == "__main__":
    main()
