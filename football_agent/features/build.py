"""Build the feature matrix and store it in DuckDB as the `features` table.

    uv run python -m football_agent.features.build

Requires the match database (`uv run python -m football_agent.data.build`).
Idempotent: the table is replaced on every run.
"""

from __future__ import annotations

import time

from football_agent.data.db import connect
from football_agent.features.pipeline import FEATURE_COLUMNS, replay


def build() -> None:
    con = connect(read_only=False)
    matches = con.sql("SELECT * FROM matches ORDER BY match_date").df()
    goals = con.sql("SELECT * FROM goals ORDER BY match_date, minute").df()

    start = time.perf_counter()
    features, _ = replay(matches, goals)
    elapsed = time.perf_counter() - start

    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features")
    con.close()

    played = int(features["is_played"].sum())
    print(
        f"features  {len(features):,} rows x {len(FEATURE_COLUMNS)} features "
        f"({played:,} played, {len(features) - played} scheduled) in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    build()
