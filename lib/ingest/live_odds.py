import os
import requests
import polars as pl
from lib.constants.nba_teams import NBA_TEAMS
from lib.utils.team_name_map import normalize_name

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise EnvironmentError("❌ Missing ODDS_API_KEY. Run: export ODDS_API_KEY='your_api_key_here'")

def fetch_live_odds():
    print("[live_odds] 🔄 Fetching DraftKings/FanDuel NBA odds...")

    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "decimal",
    }

    resp = requests.get(url, params=params, timeout=15)
    if resp.status_code != 200:
        raise ConnectionError(f"❌ API Error {resp.status_code}: {resp.text}")

    data = resp.json()
    rows = []

    for g in data:
        home, away = g.get("home_team"), g.get("away_team")
        if not home or not away:
            continue
        home, away = normalize_name(home), normalize_name(away)
        if home not in NBA_TEAMS or away not in NBA_TEAMS:
            continue

        for book in g.get("bookmakers", []):
            if book["key"] in ["draftkings", "fanduel"]:
                markets = book.get("markets", [])
                if not markets:
                    continue
                outcomes = markets[0].get("outcomes", [])
                if len(outcomes) != 2:
                    continue

                # handle odds extraction robustly
                home_price = next((o["price"] for o in outcomes if o["name"] == home), None)
                away_price = next((o["price"] for o in outcomes if o["name"] == away), None)
                if home_price is None or away_price is None:
                    continue

                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "book": book["key"],
                    "home_odds": float(home_price),
                    "away_odds": float(away_price),
                })

    df = pl.DataFrame(rows)
    df.write_parquet("data/warehouse/NBA/live_odds.parquet")
    print(f"✅ Saved {len(df)} NBA odds → data/warehouse/NBA/live_odds.parquet")

if __name__ == "__main__":
    fetch_live_odds()
