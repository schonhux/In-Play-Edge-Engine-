import sys, os
import polars as pl

# make sure root directory (project base) is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lib.constants.nba_teams import NBA_TEAMS

# Read your current messy stats file
df = pl.read_parquet("data/warehouse/NBA/current_team_stats.parquet")

# Filter to NBA-only
clean_df = df.filter(pl.col("TEAM_NAME").is_in(NBA_TEAMS))

# Save cleaned file (overwrite)
clean_df.write_parquet("data/warehouse/NBA/current_team_stats.parquet")

print(f"✅ Cleaned: {clean_df.height} rows left — NBA only.")
print("Teams included:")
print(clean_df.select('TEAM_NAME').unique())
