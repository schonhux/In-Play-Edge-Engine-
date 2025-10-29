import warnings
warnings.filterwarnings("ignore")  # suppress sklearn/lightgbm warnings

import polars as pl
import numpy as np
import joblib
from rich.console import Console
from rich.table import Table
from rich import box
from lib.modeling.utils import prob_to_moneyline

console = Console()

def calc_implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1 / decimal_odds if decimal_odds and decimal_odds > 0 else np.nan

def main(stake: float = 100):
    console.print("\n🏈 [bold bright_white]In-Play Edge Engine — Top NFL Value Bets (Live)[/bold bright_white]")
    console.print("─" * 65)

    # 🏈 Load NFL-specific model and data
    stats = pl.read_parquet("data/warehouse/NFL/current_team_stats.parquet")
    odds = pl.read_parquet("data/warehouse/NFL/live_odds.parquet")
    model = joblib.load("artifacts/NFL/model.joblib")
    cal = joblib.load("artifacts/NFL/calibrator.joblib")

    # Unique game matchups (home vs away)
    matchups = odds.unique(subset=["home_team", "away_team"])
    results = []

    for g in matchups.iter_rows(named=True):
        home, away = g["home_team"], g["away_team"]

        # Match team stats
        t1 = stats.filter(pl.col("TEAM_NAME") == home)
        t2 = stats.filter(pl.col("TEAM_NAME") == away)
        if t1.is_empty() or t2.is_empty():
            continue

        # Simple 3-feature input
        net_ypp_diff = float(t1["NET_YPP"][0] - t2["NET_YPP"][0])
        margin_diff = float(t1["PLUS_MINUS"][0] - t2["PLUS_MINUS"][0])
        ypp_diff = float(t1["YPP"][0] - t2["YPP"][0])
        X = np.array([[1.0, net_ypp_diff, margin_diff]])

        # Predict calibrated probability of home team winning
        model_prob = float(cal.predict_proba(X)[:, 1][0])

        # Pick DraftKings odds (fallback to any available)
        dk = odds.filter(
            (pl.col("home_team") == home)
            & (pl.col("away_team") == away)
            & (pl.col("book") == "draftkings")
        )
        if dk.is_empty():
            dk = odds.filter(
                (pl.col("home_team") == home)
                & (pl.col("away_team") == away)
            )
        if dk.is_empty():
            continue

        home_odds = float(dk["home_odds"][0])
        implied = calc_implied_prob(home_odds)
        edge = (model_prob - implied) * 100
        expected_value = (model_prob * home_odds - 1) * stake

        results.append({
            "matchup": f"{home} vs {away}",
            "model": model_prob,
            "implied": implied,
            "edge": edge,
            "ev": expected_value,
        })

    if not results:
        console.print("[red]No valid NFL matchups found or odds not available.[/red]\n")
        return

    df = pl.DataFrame(results).sort("edge", descending=True).head(10)

    # 🎨 Build table
    table = Table(
        show_header=True,
        header_style="bold bright_white",
        box=box.ROUNDED,
        border_style="bright_black",
    )

    table.add_column("#", justify="center")
    table.add_column("Matchup", justify="left", no_wrap=True)
    table.add_column("Model", justify="right")
    table.add_column("Impl.", justify="right")
    table.add_column("Edge%", justify="right")
    table.add_column("EV($)", justify="right")

    for i, row in enumerate(df.iter_rows(named=True), start=1):
        edge_color = "green" if row["edge"] > 0 else "red"
        ev_color = "green" if row["ev"] > 0 else "red"
        table.add_row(
            str(i),
            row["matchup"],
            f"{row['model']:.3f}",
            f"{row['implied']:.3f}",
            f"[{edge_color}]{row['edge']:+.1f}[/{edge_color}]",
            f"[{ev_color}]{row['ev']:+.2f}[/{ev_color}]",
        )

    console.print(table)
    console.print("─" * 65)
    console.print(f"🏁 [bold bright_white]Top {len(df)} live NFL opportunities | stake=${stake}[/bold bright_white]\n")

if __name__ == "__main__":
    main()
