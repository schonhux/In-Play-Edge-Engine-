from __future__ import annotations
import numpy as np, polars as pl
from pathlib import Path
from datetime import datetime, timedelta, timezone
from lib.common.settings import Settings


RNG = np.random.default_rng(7)

def simulate_game(game_id: str, start: datetime, ticks: int = 1800, dt_ms: int = 500) -> pl.DataFrame:
    base_odds = np.array([1.90, 1.90])  # HOME, AWAY
    rows = []
    ts = start
    score_h = score_a = 0
    for _ in range(ticks):
        shock = RNG.normal(0, 0.01, size=2)
        base_odds += (1.90 - base_odds) * 0.02 + shock
        base_odds = np.clip(base_odds, 1.3, 4.5)
        if RNG.random() < 0.005:
            if RNG.random() < 0.5:
                score_h += 2; base_odds[0] = max(1.2, base_odds[0]-0.1); base_odds[1] = min(6.0, base_odds[1]+0.1)
            else:
                score_a += 2; base_odds[0] = min(6.0, base_odds[0]+0.1); base_odds[1] = max(1.2, base_odds[1]-0.1)
        ts += timedelta(milliseconds=dt_ms)
        rows.append((ts, "ML", "HOME", float(base_odds[0]), score_h, score_a))
        rows.append((ts, "ML", "AWAY", float(base_odds[1]), score_h, score_a))
    return (
    pl.DataFrame(
        rows,
        schema=["ts", "market_id", "runner", "odds", "score_h", "score_a"],
        orient="row",  # <-- add this
    )
    .with_columns(
        pl.lit(game_id).alias("game_id"),
        pl.col("ts").dt.replace_time_zone("UTC"),
    )
)


def main():
    s = Settings.load()
    Path(s.raw_dir).mkdir(parents=True, exist_ok=True)
    t0 = datetime.now(timezone.utc).replace(microsecond=0)
    df = pl.concat([
        simulate_game("G001", t0 + timedelta(minutes=10)),
        simulate_game("G002", t0 + timedelta(minutes=20)),
        simulate_game("G003", t0 + timedelta(minutes=30)),
    ]).sort(["game_id","ts","runner"])
    out = Path(s.raw_dir) / "toy_ticks.parquet"
    df.write_parquet(out)
    print(f"Wrote {out} rows={df.height} games={df.select(pl.col('game_id')).n_unique()}")

if __name__ == "__main__":
    main()
