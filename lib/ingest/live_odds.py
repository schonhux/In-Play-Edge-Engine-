# lib/ingest/live_odds.py
import requests, polars as pl, os, json

API_KEY = os.getenv("ODDS_API_KEY")

def fetch_live_odds():
    if not API_KEY:
        raise ValueError("❌ Missing ODDS_API_KEY environment variable")

    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds/"
        f"?regions=us&markets=h2h&apiKey={API_KEY}"
    )

    print(f"[live_odds] Requesting odds from TheOddsAPI...")
    r = requests.get(url)

    try:
        data = r.json()
    except json.JSONDecodeError:
        print("❌ API did not return valid JSON. Raw response:")
        print(r.text)
        return

    # The API sometimes returns an error dict instead of list of games
    if isinstance(data, dict) and "message" in data:
        print("❌ API Error:", data["message"])
        return

    if not isinstance(data, list):
        print("❌ Unexpected data format:", type(data), data)
        return

    rows = []
    for g in data:
        home = g["home_team"]
        away = g["away_team"]
        for book in g.get("bookmakers", []):
            if book["key"] in ["draftkings", "fanduel"]:
                markets = book.get("markets", [])
                if not markets: continue
                outcomes = markets[0]["outcomes"]
                h_odds = next((o["price"] for o in outcomes if o["name"] == home), None)
                a_odds = next((o["price"] for o in outcomes if o["name"] == away), None)
                rows.append((home, away, book["key"], h_odds, a_odds))

    if not rows:
        print("⚠️ No odds data returned — possibly offseason or no games scheduled.")
        return

    df = pl.DataFrame(rows, schema=["home_team","away_team","book","home_odds","away_odds"])
    df.write_parquet("data/warehouse/NBA/live_odds.parquet")
    print(f"[live_odds] ✅ wrote live_odds.parquet rows={len(df)}")
    print(df.head())

if __name__ == "__main__":
    fetch_live_odds()
