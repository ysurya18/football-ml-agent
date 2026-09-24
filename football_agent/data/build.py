"""Build the DuckDB match database from the raw CSVs.

Idempotent: drops and recreates every table, so re-running after a data refresh
is safe. Run with:

    uv run python -m football_agent.data.build
"""

from __future__ import annotations

import duckdb

from football_agent.data.db import RAW_DATA_DIR, db_path
from football_agent.data.tournaments import K_FACTORS, classify

# The dataset carries scheduled-but-unplayed fixtures with "NA" scores (the 2026
# World Cup schedule). Without nullstr they break integer type inference, and
# silently dropping them would throw away the fixture list the agent predicts on.
_NULLSTR = "NA"


def _read_csv(name: str) -> str:
    """SQL expression reading one raw CSV with consistent null handling."""
    return f"read_csv('{RAW_DATA_DIR / name}', nullstr='{_NULLSTR}', header=true)"


def build() -> None:
    path = db_path()
    con = duckdb.connect(str(path))

    # --- tournament tier lookup -------------------------------------------------
    # Classification lives in Python (football_agent.data.tournaments) so the
    # feature pipeline and ad-hoc SQL share one definition. Materialise it as a
    # table rather than duplicating the rules in SQL.
    tournaments = [
        r[0]
        for r in con.sql(f"SELECT DISTINCT tournament FROM {_read_csv('results.csv')}").fetchall()
    ]
    con.execute("DROP TABLE IF EXISTS tournament_tiers")
    con.execute(
        "CREATE TABLE tournament_tiers (tournament VARCHAR PRIMARY KEY, tier VARCHAR, k_factor INTEGER)"
    )
    con.executemany(
        "INSERT INTO tournament_tiers VALUES (?, ?, ?)",
        [(t, classify(t), K_FACTORS[classify(t)]) for t in tournaments],
    )

    # --- matches ----------------------------------------------------------------
    # `outcome` is from the home team's perspective and is the model's target.
    # It is NULL for unplayed fixtures, which is what marks them non-trainable.
    con.execute("DROP TABLE IF EXISTS matches")
    con.execute(f"""
        CREATE TABLE matches AS
        SELECT
            CAST(r.date AS DATE)              AS match_date,
            r.home_team,
            r.away_team,
            r.home_score,
            r.away_score,
            r.tournament,
            t.tier                            AS tournament_tier,
            t.k_factor,
            r.city,
            r.country,
            r.neutral,
            r.home_score IS NOT NULL          AS is_played,
            CASE
                WHEN r.home_score IS NULL      THEN NULL
                WHEN r.home_score > r.away_score THEN 'home_win'
                WHEN r.home_score = r.away_score THEN 'draw'
                ELSE 'away_win'
            END                               AS outcome,
            r.home_score - r.away_score       AS goal_difference
        FROM {_read_csv("results.csv")} r
        JOIN tournament_tiers t USING (tournament)
        ORDER BY match_date, r.home_team, r.away_team
    """)

    # --- goals ------------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS goals")
    con.execute(f"""
        CREATE TABLE goals AS
        SELECT CAST(date AS DATE) AS match_date, home_team, away_team,
               team, scorer, minute, own_goal, penalty
        FROM {_read_csv("goalscorers.csv")}
        ORDER BY match_date, minute
    """)

    # --- shootouts --------------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS shootouts")
    con.execute(f"""
        CREATE TABLE shootouts AS
        SELECT CAST(date AS DATE) AS match_date, home_team, away_team, winner, first_shooter
        FROM {_read_csv("shootouts.csv")}
        ORDER BY match_date
    """)

    # --- former names -----------------------------------------------------------
    con.execute("DROP TABLE IF EXISTS former_names")
    con.execute(f"""
        CREATE TABLE former_names AS
        SELECT current, former, CAST(start_date AS DATE) AS start_date,
               CAST(end_date AS DATE) AS end_date
        FROM {_read_csv("former_names.csv")}
    """)

    _report(con)
    con.close()
    print(f"\nBuilt {path}")


def _report(con: duckdb.DuckDBPyConnection) -> None:
    played, scheduled, first, last = con.sql("""
        SELECT count(*) FILTER (WHERE is_played),
               count(*) FILTER (WHERE NOT is_played),
               min(match_date), max(match_date)
        FROM matches
    """).fetchone()
    print(f"matches        {played:>7,} played   {scheduled:>3} scheduled   {first} .. {last}")

    for table in ("goals", "shootouts", "former_names"):
        n = con.sql(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table:<15}{n:>7,}")

    print("\ntournament tiers:")
    for tier, n_t, n_m, k in con.sql("""
        SELECT t.tier, count(DISTINCT t.tournament), count(m.tournament), any_value(t.k_factor)
        FROM tournament_tiers t LEFT JOIN matches m USING (tournament)
        GROUP BY t.tier ORDER BY any_value(t.k_factor) DESC
    """).fetchall():
        print(f"  {tier:<12} K={k:<3} {n_t:>3} tournaments  {n_m:>7,} matches")


if __name__ == "__main__":
    build()
