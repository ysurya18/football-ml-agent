"""Integration tests over the built DuckDB database.

Skipped when the database has not been built yet.
"""

import pytest

from football_agent.data.db import connect, db_path

pytestmark = pytest.mark.skipif(
    not db_path().exists(),
    reason="run `uv run python -m football_agent.data.build` first",
)


@pytest.fixture(scope="module")
def con():
    c = connect()
    yield c
    c.close()


def test_expected_row_counts(con):
    played, scheduled = con.sql(
        "SELECT count(*) FILTER (WHERE is_played), count(*) FILTER (WHERE NOT is_played) FROM matches"
    ).fetchone()
    assert played == 49_215
    assert scheduled == 72


def test_scheduled_fixtures_are_the_2026_world_cup(con):
    rows = con.sql("SELECT DISTINCT tournament FROM matches WHERE NOT is_played").fetchall()
    assert rows == [("FIFA World Cup",)]


def test_unplayed_fixtures_have_no_outcome(con):
    """The training set is defined by outcome IS NOT NULL, so this is the guard
    that keeps unplayed fixtures out of the model."""
    leaked = con.sql(
        "SELECT count(*) FROM matches WHERE NOT is_played AND outcome IS NOT NULL"
    ).fetchone()[0]
    assert leaked == 0


def test_played_matches_always_have_an_outcome(con):
    missing = con.sql(
        "SELECT count(*) FROM matches WHERE is_played AND outcome IS NULL"
    ).fetchone()[0]
    assert missing == 0


def test_outcome_matches_the_score(con):
    wrong = con.sql("""
        SELECT count(*) FROM matches WHERE is_played AND outcome <> CASE
            WHEN home_score > away_score THEN 'home_win'
            WHEN home_score = away_score THEN 'draw' ELSE 'away_win' END
    """).fetchone()[0]
    assert wrong == 0


def test_world_cup_tier_excludes_qualifiers(con):
    """1,036 finals matches, not the 9,807 you get by counting qualifiers too."""
    n = con.sql("SELECT count(*) FROM matches WHERE tournament_tier = 'world_cup'").fetchone()[0]
    assert n == 1_036


def test_copa_america_is_continental(con):
    """Covers both upstream defects at once: the accented name resolves to a
    continental championship, while its qualifier still resolves to a qualifier."""
    tiers = dict(
        con.sql("""
        SELECT tournament, any_value(tournament_tier) FROM matches
        WHERE tournament LIKE 'Copa Am%' GROUP BY tournament
    """).fetchall()
    )
    assert tiers == {
        "Copa América": "continental",
        "Copa América qualification": "qualifier",
    }


def test_matches_are_chronologically_ordered(con):
    """The feature pipeline replays this table in order; unordered rows would
    silently leak future results into past feature rows."""
    out_of_order = con.sql("""
        SELECT count(*) FROM (
            SELECT match_date, lag(match_date) OVER () AS prev FROM matches
        ) WHERE prev IS NOT NULL AND match_date < prev
    """).fetchone()[0]
    assert out_of_order == 0


def test_no_null_team_names(con):
    n = con.sql(
        "SELECT count(*) FROM matches WHERE home_team IS NULL OR away_team IS NULL"
    ).fetchone()[0]
    assert n == 0


def test_every_match_has_a_k_factor(con):
    n = con.sql("SELECT count(*) FROM matches WHERE k_factor IS NULL").fetchone()[0]
    assert n == 0


def test_every_tier_key_matches_a_real_tournament(con):
    """A key that matches nothing is a silent misclassification: the tournament
    it was meant for falls through to the default tier. 'CONCACAF Gold Cup' did
    exactly that — the data calls it 'Gold Cup'."""
    from football_agent.data.tournaments import _CONTINENTAL, _NATIONS_LEAGUE, _WORLD_CUP, _fold

    names = [_fold(r[0]) for r in con.sql("SELECT DISTINCT tournament FROM matches").fetchall()]
    dead = [
        k for k in (*_CONTINENTAL, _WORLD_CUP, _NATIONS_LEAGUE) if not any(k in n for n in names)
    ]
    assert dead == []


def test_gold_cup_is_continental(con):
    tier = con.sql("SELECT any_value(tournament_tier) FROM matches WHERE tournament = 'Gold Cup'")
    assert tier.fetchone()[0] == "continental"
