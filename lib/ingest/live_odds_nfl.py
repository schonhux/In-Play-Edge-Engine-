import os
import requests
import polars as pl

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

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[live_odds_nfl] ⚠️ Failed to fetch live data: {e}")
        data = []

    rows = []
    for g in data:
        home = g.get("home_team")
        away = g.get("away_team")
        if not home or not away:
            continue

        for book in g.get("bookmakers", []):
            if book["key"] in ["draftkings", "fanduel"]:
                markets = book.get("markets", [])
                if not markets:
                    continue
                outcomes = markets[0].get("outcomes", [])
                if len(outcomes) != 2:
                    continue

                home_price = next((o["price"] for o in outcomes if o["name"] == home), None)
                away_price = next((o["price"] for o in outcomes if o["name"] == away), None)
                if home_price is None or away_price is None:
                    continue

                rows.append({
                    "home_team": home,
                    "away_team": away,
                    "book": book["key"],
                    "home_odds": home_price,
                    "away_odds": away_price,
                })

    if not rows:
        print("[live_odds_nfl] ⚠️ No live NFL odds found. Using fallback mock data...")
        rows = [
            {"home_team": "Kansas City Chiefs", "away_team": "Buffalo Bills", "book": "mock", "home_odds": 1.91, "away_odds": 1.91},
            {"home_team": "Dallas Cowboys", "away_team": "San Francisco 49ers", "book": "mock", "home_odds": 2.05, "away_odds": 1.80},
            {"home_team": "Miami Dolphins", "away_team": "New England Patriots", "book": "mock", "home_odds": 1.87, "away_odds": 2.10},
            {"home_team": "Chicago Bears", "away_team": "Green Bay Packers", "book": "mock", "home_odds": 2.30, "away_odds": 1.68},
            {"home_team": "New York Jets", "away_team": "Cleveland Browns", "book": "mock", "home_odds": 1.95, "away_odds": 1.95},
            {"home_team": "Detroit Lions", "away_team": "Los Angeles Rams", "book": "mock", "home_odds": 2.00, "away_odds": 1.90},
            {"home_team": "Philadelphia Eagles", "away_team": "Seattle Seahawks", "book": "mock", "home_odds": 1.85, "away_odds": 2.15},
            {"home_team": "Tennessee Titans", "away_team": "Baltimore Ravens", "book": "mock", "home_odds": 2.40, "away_odds": 1.65},
        ]

    df = pl.DataFrame(rows)
    df.write_parquet("data/warehouse/NFL/live_odds.parquet")
    print(f"✅ Saved {len(df)} NFL odds → data/warehouse/NFL/live_odds.parquet")

if __name__ == "__main__":
    fetch_live_odds()

