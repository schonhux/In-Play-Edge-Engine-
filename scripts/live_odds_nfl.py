import os
import requests
import polars as pl
from lib.constants.nfl_teams import NFL_TEAMS

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise EnvironmentError("❌ Missing ODDS_API_KEY. Run: export ODDS_API_KEY='your_api_key_here'")

def fetch_live_odds():
    print("[live_odds_nfl] 🔄 Fetching DraftKings/FanDuel NFL odds...")

    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }

    resp = requests.get(url, params=params)
    data = resp.json()

    rows = []
    for g in data:
        home = g.get("home_team")
        away = g.get("away_team")
        if home not in NFL_TEAMS or away not in NFL_TEAMS:
            continue

        for book in g.get("bookmakers", []):
            if book["key"] in ["draftkings", "fanduel"]:
                markets = book.get("markets", [])
                if not markets:
                    continue
                outcomes = markets[0].get("outcomes", [])
                if len(outcomes) != 2:
                    continue
                home_price = outcomes[0]["price"] if outcomes[0]["name"] == home else outcomes[1]["price"]
                away_price = outcomes[1]["price"] if outcomes[0]["name"] == home else outcomes[0]["price"]

                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "book": book["key"],
                    "home_odds": home_price,
                    "away_odds": away_price,
                })

    df = pl.DataFrame(rows)
    df.write_parquet("data/warehouse/NFL/live_odds.parquet")
    print(f"✅ Saved {len(df)} NFL odds → data/warehouse/NFL/live_odds.parquet")

if __name__ == "__main__":
    fetch_live_odds()
