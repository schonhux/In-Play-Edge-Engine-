from nba_api.stats.endpoints import leaguedashteamstats
import polars as pl

def fetch_nba_team_stats():
    stats = leaguedashteamstats.LeagueDashTeamStats(
        season='2024-25',
        season_type_all_star='Regular Season'
    ).get_data_frames()[0]

    stats = stats[[
        "TEAM_ID","TEAM_NAME","GP","FG_PCT","REB","AST","PLUS_MINUS"
    ]]

    df = pl.from_pandas(stats)
    df.write_parquet("data/warehouse/NBA/current_team_stats.parquet")
    print("[nba_current_stats] ✅ wrote current_team_stats.parquet rows=", len(df))

if __name__ == "__main__":
    fetch_nba_team_stats()
