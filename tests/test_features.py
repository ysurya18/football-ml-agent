"""Tests for the feature pipeline.

The leakage tests come first because they guard the property everything else
depends on: a match's feature row is built only from matches before it.
"""

import ast
import pickle
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from football_agent.features import trackers as t
from football_agent.features.pipeline import (
    FEATURE_COLUMNS,
    FEATURE_FAMILIES,
    FeatureState,
    MatchContext,
    replay,
)
from football_agent.features.venue import altitude, derive_confederations, venue_features

TEAMS = ["Alpha", "Beta", "Gamma", "Delta"]


def _matches(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """A synthetic round-robin with deterministic pseudo-random scores."""
    rows = []
    for i in range(n):
        home, away = TEAMS[i % 4], TEAMS[(i + 1 + i // 4) % 4]
        if home == away:
            away = TEAMS[(i + 2) % 4]
        rows.append(
            {
                "match_date": date(2000, 1, 1) + timedelta(days=7 * i),
                "home_team": home,
                "away_team": away,
                "home_score": (i * 7 + seed) % 4,
                "away_score": (i * 3 + seed) % 3,
                "tournament": ["Friendly", "FIFA World Cup", "UEFA Euro qualification"][i % 3],
                "tournament_tier": None,
                "city": ["Bogotá", "London", None][i % 3],
                "country": None,
                "neutral": i % 5 == 0,
                "is_played": True,
                "outcome": None,
            }
        )
    return pd.DataFrame(rows)


def _goals(matches: pd.DataFrame) -> pd.DataFrame:
    """One timed goal row per goal, home goals first."""
    rows = []
    for m in matches[matches.is_played].itertuples():
        minute = 10
        for side, n in ((m.home_team, m.home_score), (m.away_team, m.away_score)):
            for _ in range(int(n)):
                rows.append(
                    {
                        "match_date": m.match_date,
                        "home_team": m.home_team,
                        "away_team": m.away_team,
                        "team": side,
                        "scorer": f"{side}-{minute % 3}",
                        "minute": minute,
                        "own_goal": False,
                        "penalty": minute % 20 == 0,
                    }
                )
                minute += 10
    return pd.DataFrame(rows)


def _replay(matches):
    return replay(matches, _goals(matches))[0][FEATURE_COLUMNS]


# --- leakage ----------------------------------------------------------------------


class TestNoLeakage:
    @pytest.mark.parametrize("k", [10, 30, 59])
    def test_row_ignores_its_own_and_every_later_result(self, k):
        base = _matches()
        altered = base.copy()
        altered.loc[k:, "home_score"] += 3
        altered.loc[k:, "away_score"] = 0

        a, b = _replay(base), _replay(altered)

        pd.testing.assert_frame_equal(a.iloc[: k + 1], b.iloc[: k + 1])
        # ...and the alteration does reach later rows, so the test has teeth.
        if k + 1 < len(base):
            assert not a.iloc[k + 1 :].equals(b.iloc[k + 1 :])

    def test_unplayed_fixture_does_not_update_state(self):
        base = _matches(40)
        with_fixture = base.copy()
        with_fixture.loc[20, ["home_score", "away_score"]] = None
        with_fixture.loc[20, "is_played"] = False

        a, b = _replay(base), _replay(with_fixture)
        # The fixture itself still gets a (pre-match) row...
        pd.testing.assert_series_equal(a.iloc[20], b.iloc[20])
        # ...but later rows differ, because its result was never learned.
        assert not a.iloc[21:].equals(b.iloc[21:])

    def test_features_do_not_mutate_state(self):
        _, state = replay(_matches(), _goals(_matches()))
        before = pickle.dumps(state)
        ctx = MatchContext(date=date(2030, 1, 1), tournament="FIFA World Cup", city="Quito")
        state.features("Alpha", "Beta", ctx)
        state.features("Alpha", "Unseen Team", ctx)
        assert pickle.dumps(state) == before

    def test_update_rejects_out_of_order_matches(self):
        state = FeatureState({})
        state.update("Alpha", "Beta", MatchContext(date=date(2000, 2, 1)), 1, 0)
        with pytest.raises(ValueError):
            state.update("Alpha", "Beta", MatchContext(date=date(2000, 1, 1)), 1, 0)


# --- shape --------------------------------------------------------------------------


class TestColumns:
    def test_92_unique_features(self):
        assert len(FEATURE_COLUMNS) == 92
        assert len(set(FEATURE_COLUMNS)) == 92

    def test_same_names_and_order_as_upstream(self):
        """Parsed, not imported, so the vendored module never executes."""
        src = Path("reference/enhanced_features_upstream.py").read_text()
        lists = {
            node.targets[0].id: ast.literal_eval(node.value)
            for node in ast.parse(src).body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.List)
        }
        upstream = [
            f
            for name in (
                "ORIGINAL_FEATURES",
                "NEW_GOALSCORER_FEATURES",
                "NEW_MOMENTUM_FEATURES",
                "NEW_POISSON_FEATURES",
                "NEW_VENUE_FEATURES",
                "NEW_TOURNAMENT_FEATURES",
            )
            for f in lists[name]
        ]
        assert FEATURE_COLUMNS == upstream

    def test_features_returns_every_column_in_order(self):
        row = FeatureState({}).features("A", "B", MatchContext(date=date(2000, 1, 1)))
        assert list(row) == FEATURE_COLUMNS

    def test_families_partition_the_columns(self):
        assert sum(len(f) for f in FEATURE_FAMILIES.values()) == 92


# --- trackers ------------------------------------------------------------------------


class TestElo:
    def test_first_home_win_by_hand(self):
        """1500 v 1500 with +100 home advantage: expected 0.640. A 1-0 friendly
        win (K=20, margin multiplier 1) moves the ratings by 20 * 0.360."""
        elo = t.Elo()
        exp = 1 / (1 + 10 ** (-100 / 400))
        assert elo.features("A", "B", "Friendly", False)["home_expected"] == pytest.approx(exp)
        elo.update("A", "B", 1, 0, "Friendly", False)
        assert elo.rating("A") == pytest.approx(1500 + 20 * (1 - exp))
        assert elo.rating("B") == pytest.approx(1500 - 20 * (1 - exp))

    def test_zero_sum(self):
        elo = t.Elo()
        elo.update("A", "B", 4, 1, "FIFA World Cup", True)
        assert elo.rating("A") + elo.rating("B") == pytest.approx(3000)

    def test_tier_rating_only_moves_in_its_tier(self):
        elo = t.Elo()
        elo.update("A", "B", 2, 0, "FIFA World Cup", True)
        assert elo.tier_rating("world_cup", "A") > 1500
        assert elo.tier_rating("friendly", "A") == 1500

    def test_qualifier_uses_qualifier_k(self):
        """Carries the tournament-tier fix through to Elo: a qualifier moves
        ratings at K=40, not the World Cup's K=60."""
        wc, q = t.Elo(), t.Elo()
        wc.update("A", "B", 1, 0, "FIFA World Cup", True)
        q.update("A", "B", 1, 0, "FIFA World Cup qualification", True)
        assert (q.rating("A") - 1500) / (wc.rating("A") - 1500) == pytest.approx(40 / 60)


class TestForm:
    def test_defaults_until_three_matches(self):
        f = t.TeamForm()
        f.update(date(2000, 1, 1), 3, 0)
        f.update(date(2000, 1, 8), 3, 0)
        assert f.form(5) == 0.5
        f.update(date(2000, 1, 15), 0, 0)
        assert f.form(5) == pytest.approx(2.5 / 3)

    def test_weighted_form_favours_recent(self):
        f = t.TeamForm()
        for gf, ga in ((0, 1), (0, 1), (1, 0)):  # loss, loss, win
            f.update(date(2000, 1, 1), gf, ga)
        assert f.weighted_form(10) == pytest.approx(1 / (0.81 + 0.9 + 1))

    def test_days_rest(self):
        f = t.TeamForm()
        assert f.days_rest(date(2000, 1, 1)) == 30
        f.update(date(2000, 1, 1), 1, 1)
        assert f.days_rest(date(2000, 1, 11)) == 10


class TestHeadToHead:
    def test_perspective(self):
        h = t.HeadToHead()
        h.update("A", "B", 2, 0)
        h.update("B", "A", 1, 1)
        assert h.features("A", "B") == {"h2h_win_rate": 0.5, "h2h_matches": 2, "h2h_goal_diff": 1.0}
        assert h.features("B", "A")["h2h_goal_diff"] == -1.0


class TestConcededFirst:
    def test_own_goal_opener_counts_for_the_benefiting_side(self):
        """Regression: `Goal.team` is already the side the goal counts for.
        Upstream inverted own goals again and credited the opener to the wrong
        side in 332 matches."""
        goals = [
            t.Goal("A", "B defender", 5, own_goal=True, penalty=False),
            t.Goal("B", "B striker", 60, own_goal=False, penalty=False),
            t.Goal("A", "A striker", 80, own_goal=False, penalty=False),
        ]
        assert t.conceded_first("A", 2, 1, goals) is False
        assert t.conceded_first("B", 1, 2, goals) is True

    def test_unknown_without_goal_data(self):
        """Regression: upstream fell back to 'conceded at all', so a 2-1 win
        with no goal rows was always scored as a comeback."""
        assert t.conceded_first("A", 2, 1, []) is None

    def test_known_without_goal_data_when_one_side_did_not_score(self):
        assert t.conceded_first("A", 1, 0, []) is False
        assert t.conceded_first("A", 0, 2, []) is True

    def test_comeback_rate_ignores_unknown_matches(self):
        m = t.Momentum()
        m.update("A", 2, 1, True)  # came from behind
        for _ in range(4):
            m.update("A", 2, 1, None)  # unknown order: not evidence either way
        assert m.features("A")["comeback_rate"] == 1.0


class TestMomentum:
    def test_streaks(self):
        m = t.Momentum()
        for gf, ga in [(1, 0), (0, 1), (1, 1), (2, 0), (3, 0)]:
            m.update("A", gf, ga, None)
        f = m.features("A")
        assert f["current_streak"] == 2
        assert f["unbeaten_streak"] == 3
        assert f["blowout_win_pct"] == pytest.approx(1 / 5)


class TestPoisson:
    def _tracker(self, scored, conceded):
        p = t.PoissonGoals()
        for team in ("A", "B"):
            for s, c in zip(scored, conceded):
                p.update(team, s, c)
        return p

    def test_probabilities_are_sane(self):
        f = self._tracker([1, 2, 0, 3, 1], [1, 0, 2, 1, 1]).features("A", "B")
        assert 0 < f["home_poisson_win"] < 1
        # Identical teams: win and loss equally likely.
        loss = 1 - f["home_poisson_win"] - f["home_poisson_draw"]
        assert f["home_poisson_win"] == pytest.approx(loss, abs=1e-9)

    def test_order_independent(self):
        """Regression: upstream's PMF cache made a value depend on which lambda
        had filled the cache bucket first."""
        p = self._tracker([1, 2, 0, 3, 1], [1, 0, 2, 1, 1])
        q = self._tracker([4, 4, 3, 5, 4], [0, 0, 1, 0, 0])
        first = p.features("A", "B")
        q.features("A", "B")
        assert p.features("A", "B") == first

    def test_high_scoring_mass_is_not_truncated(self):
        pmf = t._poisson_pmf(5.0, t.PoissonGoals.MAX_GOALS)
        assert sum(pmf) > 0.9999


class TestStage:
    @pytest.mark.parametrize(
        "tournament,expected",
        [
            ("FIFA World Cup", "wc_finals"),
            ("FIFA World Cup qualification", "qualifying"),
            ("Copa América", "continental_finals"),
            ("UEFA Euro", "continental_finals"),
            ("Friendly", "friendly"),
            # Regression: upstream's "copa"/"euro" substrings made these finals.
            ("Copa Newton", "other_competitive"),
            ("Central European International Cup", "other_competitive"),
            ("CONIFA European Football Cup", "other_competitive"),
        ],
    )
    def test_stage(self, tournament, expected):
        assert t.stage(tournament) == expected


# --- venue ---------------------------------------------------------------------------


class TestVenue:
    def test_accented_city_names(self):
        """Regression: upstream's 'bogota' never matched the data's 'Bogotá'."""
        assert altitude("Bogotá") == altitude("Bogota") == 2640
        assert altitude("São Paulo") == 760

    def test_unknown_city_default(self):
        assert altitude("Nowhere") == altitude(None) == 100

    def test_unknown_teams_are_not_the_same_confederation(self):
        """Regression: upstream mapped both to 'OTHER' and called them equal."""
        f = venue_features("Jersey", "Padania", "London", {})
        assert f["same_confederation"] == 0
        assert f["is_intercontinental"] == 1

    def test_derived_confederation_outvotes_guest_appearances(self):
        m = pd.DataFrame(
            {
                "tournament": ["Copa América"] * 2 + ["Gold Cup qualification"] * 3,
                "home_team": ["Mexico"] * 5,
                "away_team": ["Brazil", "Chile", "Honduras", "Panama", "Jamaica"],
            }
        )
        c = derive_confederations(m)
        assert c["Mexico"] == "CONCACAF"
        assert c["Brazil"] == "CONMEBOL"

    def test_derivation_ignores_non_confederation_tournaments(self):
        m = pd.DataFrame({"tournament": ["Friendly"], "home_team": ["X"], "away_team": ["Y"]})
        assert derive_confederations(m) == {}


# --- scenarios -----------------------------------------------------------------------


@pytest.fixture(scope="module")
def state():
    return replay(_matches(), _goals(_matches()))[1]


@pytest.fixture
def ctx():
    return MatchContext(date=date(2030, 6, 1), tournament="Friendly", neutral=True)


class TestScenarios:
    def _changed(self, a, b):
        return {k for k in a if a[k] != b[k]}

    def test_altitude_changes_only_venue_altitude(self, state, ctx):
        a = state.features("Alpha", "Beta", ctx)
        b = state.features("Alpha", "Beta", replace(ctx, city="La Paz"))
        assert self._changed(a, b) == {"altitude", "is_high_altitude"}
        assert b["is_high_altitude"] == 1

    def test_altitude_override_beats_city(self, state, ctx):
        f = state.features("Alpha", "Beta", replace(ctx, city="La Paz", altitude=10))
        assert f["altitude"] == 10 and f["is_high_altitude"] == 0

    def test_home_advantage(self, state, ctx):
        a = state.features("Alpha", "Beta", ctx)
        b = state.features("Alpha", "Beta", replace(ctx, neutral=False))
        assert self._changed(a, b) == {"is_neutral", "is_home", "home_expected"}
        assert b["home_expected"] > a["home_expected"]

    def test_tournament_changes_tier_features(self, state, ctx):
        a = state.features("Alpha", "Beta", ctx)
        b = state.features("Alpha", "Beta", replace(ctx, tournament="FIFA World Cup"))
        changed = self._changed(a, b)
        assert {"is_world_cup", "is_friendly"} <= changed
        assert changed <= {"is_world_cup", "is_friendly"} | set(FEATURE_FAMILIES["elo"])

    def test_swapping_sides_at_neutral_venue_mirrors_elo(self, state, ctx):
        a = state.features("Alpha", "Beta", ctx)
        b = state.features("Beta", "Alpha", ctx)
        assert a["elo_diff"] == pytest.approx(-b["elo_diff"])
        assert a["home_expected"] == pytest.approx(1 - b["home_expected"])

    def test_unseen_team_gets_defaults(self, state, ctx):
        f = state.features("Alpha", "Brand New FC", ctx)
        assert f["away_elo"] == 1500
        assert f["away_experience"] == 0
        assert f["away_form_10"] == 0.5


# --- the real data -------------------------------------------------------------------


@pytest.fixture(scope="module")
def real():
    from football_agent.data.db import connect, db_path

    if not db_path().exists():
        pytest.skip("run `uv run python -m football_agent.data.build` first")
    con = connect()
    matches = con.sql("SELECT * FROM matches").df()
    goals = con.sql("SELECT * FROM goals").df()
    con.close()
    return replay(matches, goals)


class TestRealData:
    def test_one_row_per_match(self, real):
        features, _ = real
        assert len(features) == 49_287

    def test_scheduled_fixtures_get_rows_but_no_outcome(self, real):
        features, _ = real
        scheduled = features[~features.is_played]
        assert len(scheduled) == 72
        assert scheduled.outcome.isna().all()
        assert scheduled[FEATURE_COLUMNS].notna().all().all()

    def test_no_missing_feature_values(self, real):
        features, _ = real
        assert features[FEATURE_COLUMNS].notna().all().all()

    def test_accented_cities_resolve(self, real):
        features, _ = real
        assert (features.loc[features.city == "Bogotá", "altitude"] == 2640).all()

    def test_confederations_cover_modern_football(self, real):
        """Upstream's hand-written map left 64% of post-1990 matches with at
        least one team unmapped."""
        features, state = real
        modern = features[features.match_date >= date(1990, 1, 1)]
        known = modern.home_team.isin(state.confederations) & modern.away_team.isin(
            state.confederations
        )
        assert known.mean() > 0.95

    def test_final_state_serves_a_prediction(self, real):
        _, state = real
        f = state.features("Spain", "Brazil", MatchContext(date=date(2026, 6, 1)))
        assert f["home_experience"] > 500 and f["away_experience"] > 500
