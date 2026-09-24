"""Chronological replay: one 92-feature row per match, built from pre-match state.

    state = FeatureState(confederations)
    for match in matches (in date order):
        row   = state.features(home, away, context)   # read: pre-match only
        if match was played:
            state.update(home, away, context, score, goals)   # then learn from it

`features()` never mutates, so a row cannot see its own result, and it is the
same function whether it is building a training row or answering "who wins
Spain v Brazil in Bogotá next June?" — scenarios are a different MatchContext
passed to the post-replay state.

Unplayed fixtures get a feature row (that is what the agent predicts on) but
are never fed to update(): they have no score.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import pandas as pd

from football_agent.data.tournaments import classify
from football_agent.features import trackers as t
from football_agent.features.venue import derive_confederations, venue_features

#: Feature names by family, in upstream's column order (ALL_FEATURES in
#: reference/enhanced_features_upstream.py). The families map onto the agent's
#: per-family tools and group SHAP attributions for explanations.
FEATURE_FAMILIES: dict[str, list[str]] = {
    "elo": [
        "home_elo", "away_elo", "elo_diff", "elo_total",
        "home_tournament_elo", "away_tournament_elo", "tournament_elo_diff",
        "home_expected",
    ],
    "form": [
        "home_form_5", "home_form_10", "home_form_20",
        "away_form_5", "away_form_10", "away_form_20",
        "home_weighted_form_10", "away_weighted_form_10",
        "form_diff_5", "form_diff_10", "weighted_form_diff",
    ],
    "goals": [
        "home_goals_scored_avg_10", "home_goals_conceded_avg_10",
        "away_goals_scored_avg_10", "away_goals_conceded_avg_10",
        "home_goal_diff_avg_10", "away_goal_diff_avg_10",
        "goal_diff_differential", "attack_vs_defense",
    ],
    "h2h": ["h2h_win_rate", "h2h_matches", "h2h_goal_diff"],
    "context": [
        "is_neutral", "is_home", "is_world_cup", "is_continental", "is_friendly",
        "home_days_rest", "away_days_rest", "rest_diff",
        "home_experience", "away_experience",
    ],
    "goalscorer": [
        "home_scoring_depth", "away_scoring_depth", "scoring_depth_diff",
        "home_star_dependency", "away_star_dependency",
        "home_penalty_ratio", "away_penalty_ratio",
        "home_late_goal_ratio", "away_late_goal_ratio", "late_goal_diff",
        "home_first_half_ratio", "away_first_half_ratio",
    ],
    "momentum": [
        "home_streak", "away_streak", "streak_diff",
        "home_unbeaten", "away_unbeaten",
        "home_clean_sheet_pct", "away_clean_sheet_pct",
        "home_comeback_rate", "away_comeback_rate",
        "home_draw_tendency", "away_draw_tendency", "draw_tendency_sum",
        "home_blowout_win_pct", "away_blowout_loss_pct",
        "home_shutout_loss_pct", "away_shutout_loss_pct",
    ],
    "poisson": [
        "home_lambda", "away_lambda",
        "home_poisson_win", "home_poisson_draw",
        "home_scoring_variance", "away_scoring_variance",
        "home_overperformance", "away_overperformance",
    ],
    "venue": [
        "altitude", "is_high_altitude",
        "same_confederation", "confed_strength_diff", "is_intercontinental",
    ],
    "tournament": [
        "home_wc_form", "away_wc_form", "wc_form_diff",
        "home_competitive_form", "away_competitive_form",
        "home_big_game_factor", "away_big_game_factor", "big_game_diff",
        "home_wc_experience", "away_wc_experience", "wc_experience_diff",
    ],
}  # fmt: skip

FEATURE_COLUMNS: list[str] = [f for fam in FEATURE_FAMILIES.values() for f in fam]

#: Carried alongside the features in every row; never fed to the model.
META_COLUMNS: list[str] = [
    "match_date", "home_team", "away_team", "tournament", "tournament_tier",
    "city", "country", "neutral", "is_played", "home_score", "away_score", "outcome",
]  # fmt: skip


@dataclass(frozen=True)
class MatchContext:
    """Everything about a fixture except who plays and the result.

    For a scenario, change fields with dataclasses.replace(). `altitude`
    overrides the city lookup, for "what if it were played at 2,600 m?".
    """

    date: date
    tournament: str = "Friendly"
    neutral: bool = True
    city: str | None = None
    country: str | None = None
    altitude: float | None = None


class FeatureState:
    """All running trackers. Read with features(), advance with update()."""

    def __init__(self, confederations: dict[str, str]) -> None:
        self.confederations = confederations
        self.elo = t.Elo()
        self.form: defaultdict[str, t.TeamForm] = defaultdict(t.TeamForm)
        self.h2h = t.HeadToHead()
        self.goalscorers = t.Goalscorers()
        self.momentum = t.Momentum()
        self.poisson = t.PoissonGoals()
        self.tournaments = t.TournamentContext()
        self.last_date: date | None = None

    def _form(self, team: str) -> t.TeamForm:
        # .get, not [], so that reading an unseen team doesn't insert it.
        return self.form.get(team) or t.TeamForm()

    def features(self, home: str, away: str, ctx: MatchContext) -> dict[str, float]:
        """The 92 features for `home` v `away` in `ctx`, from the current state.

        Pure: calling it any number of times leaves the state unchanged.
        """
        tier = classify(ctx.tournament)
        hf, af = self._form(home), self._form(away)
        hg, ag = self.goalscorers.features(home), self.goalscorers.features(away)
        hm, am = self.momentum.features(home), self.momentum.features(away)
        ht, at = self.tournaments.features(home), self.tournaments.features(away)
        h_rest, a_rest = hf.days_rest(ctx.date), af.days_rest(ctx.date)

        f: dict[str, float] = {}
        f.update(self.elo.features(home, away, ctx.tournament, ctx.neutral))
        f.update(
            home_form_5=hf.form(5),
            home_form_10=hf.form(10),
            home_form_20=hf.form(20),
            away_form_5=af.form(5),
            away_form_10=af.form(10),
            away_form_20=af.form(20),
            home_weighted_form_10=hf.weighted_form(10),
            away_weighted_form_10=af.weighted_form(10),
            form_diff_5=hf.form(5) - af.form(5),
            form_diff_10=hf.form(10) - af.form(10),
            weighted_form_diff=hf.weighted_form(10) - af.weighted_form(10),
            home_goals_scored_avg_10=hf.goals_scored(10),
            home_goals_conceded_avg_10=hf.goals_conceded(10),
            away_goals_scored_avg_10=af.goals_scored(10),
            away_goals_conceded_avg_10=af.goals_conceded(10),
            home_goal_diff_avg_10=hf.goal_diff(10),
            away_goal_diff_avg_10=af.goal_diff(10),
            goal_diff_differential=hf.goal_diff(10) - af.goal_diff(10),
            attack_vs_defense=hf.goals_scored(10) - af.goals_conceded(10),
        )
        f.update(self.h2h.features(home, away))
        f.update(
            is_neutral=int(ctx.neutral),
            is_home=int(not ctx.neutral),
            is_world_cup=int(tier == "world_cup"),
            is_continental=int(tier == "continental"),
            is_friendly=int(tier == "friendly"),
            home_days_rest=h_rest,
            away_days_rest=a_rest,
            rest_diff=h_rest - a_rest,
            home_experience=hf.n_matches,
            away_experience=af.n_matches,
        )
        f.update(
            home_scoring_depth=hg["scoring_depth"],
            away_scoring_depth=ag["scoring_depth"],
            scoring_depth_diff=hg["scoring_depth"] - ag["scoring_depth"],
            home_star_dependency=hg["star_dependency"],
            away_star_dependency=ag["star_dependency"],
            home_penalty_ratio=hg["penalty_ratio"],
            away_penalty_ratio=ag["penalty_ratio"],
            home_late_goal_ratio=hg["late_goal_ratio"],
            away_late_goal_ratio=ag["late_goal_ratio"],
            late_goal_diff=hg["late_goal_ratio"] - ag["late_goal_ratio"],
            home_first_half_ratio=hg["first_half_ratio"],
            away_first_half_ratio=ag["first_half_ratio"],
        )
        f.update(
            home_streak=hm["current_streak"],
            away_streak=am["current_streak"],
            streak_diff=hm["current_streak"] - am["current_streak"],
            home_unbeaten=hm["unbeaten_streak"],
            away_unbeaten=am["unbeaten_streak"],
            home_clean_sheet_pct=hm["clean_sheet_pct"],
            away_clean_sheet_pct=am["clean_sheet_pct"],
            home_comeback_rate=hm["comeback_rate"],
            away_comeback_rate=am["comeback_rate"],
            home_draw_tendency=hm["draw_tendency"],
            away_draw_tendency=am["draw_tendency"],
            draw_tendency_sum=hm["draw_tendency"] + am["draw_tendency"],
            home_blowout_win_pct=hm["blowout_win_pct"],
            away_blowout_loss_pct=am["blowout_loss_pct"],
            home_shutout_loss_pct=hm["shutout_loss_pct"],
            away_shutout_loss_pct=am["shutout_loss_pct"],
        )
        f.update(self.poisson.features(home, away))
        f.update(venue_features(home, away, ctx.city, self.confederations, ctx.altitude))
        f.update(
            home_wc_form=ht["wc_form"],
            away_wc_form=at["wc_form"],
            wc_form_diff=ht["wc_form"] - at["wc_form"],
            home_competitive_form=ht["competitive_form"],
            away_competitive_form=at["competitive_form"],
            home_big_game_factor=ht["big_game_factor"],
            away_big_game_factor=at["big_game_factor"],
            big_game_diff=ht["big_game_factor"] - at["big_game_factor"],
            home_wc_experience=ht["wc_experience"],
            away_wc_experience=at["wc_experience"],
            wc_experience_diff=ht["wc_experience"] - at["wc_experience"],
        )
        return {name: f[name] for name in FEATURE_COLUMNS}

    def update(
        self,
        home: str,
        away: str,
        ctx: MatchContext,
        home_score: int,
        away_score: int,
        goals: list[t.Goal] = (),
    ) -> None:
        """Fold one finished match into every tracker."""
        if self.last_date is not None and ctx.date < self.last_date:
            raise ValueError(f"match on {ctx.date} replayed after {self.last_date}")
        self.last_date = ctx.date
        hs, as_ = home_score, away_score

        self.elo.update(home, away, hs, as_, ctx.tournament, ctx.neutral)
        self.form[home].update(ctx.date, hs, as_)
        self.form[away].update(ctx.date, as_, hs)
        self.h2h.update(home, away, hs, as_)
        self.poisson.update(home, hs, as_)
        self.poisson.update(away, as_, hs)
        self.tournaments.update(home, ctx.tournament, hs, as_)
        self.tournaments.update(away, ctx.tournament, as_, hs)
        self.goalscorers.update(list(goals))
        self.momentum.update(home, hs, as_, t.conceded_first(home, hs, as_, goals))
        self.momentum.update(away, as_, hs, t.conceded_first(away, as_, hs, goals))


def _as_dates(col: pd.Series) -> pd.Series:
    """datetime.date values, so replayed rows and scenarios do the same date
    arithmetic (DuckDB hands back pandas Timestamps)."""
    return pd.to_datetime(col).dt.date


def _index_goals(goals: pd.DataFrame) -> dict[tuple, list[t.Goal]]:
    goals = goals.assign(match_date=_as_dates(goals["match_date"]))
    index: dict[tuple, list[t.Goal]] = defaultdict(list)
    for r in goals.itertuples(index=False):
        index[(r.match_date, r.home_team, r.away_team)].append(
            t.Goal(
                team=r.team,
                scorer=r.scorer if isinstance(r.scorer, str) else "Unknown",
                minute=None if pd.isna(r.minute) else int(r.minute),
                own_goal=bool(r.own_goal),
                penalty=bool(r.penalty),
            )
        )
    return index


def replay(matches: pd.DataFrame, goals: pd.DataFrame) -> tuple[pd.DataFrame, FeatureState]:
    """Replay every match in date order.

    Returns the feature matrix (META_COLUMNS + FEATURE_COLUMNS, one row per
    match, unplayed fixtures included) and the final state, from which
    predictions and scenarios are served.
    """
    matches = (
        matches.assign(match_date=_as_dates(matches["match_date"]))
        .sort_values("match_date", kind="stable")
        .reset_index(drop=True)
    )
    goal_index = _index_goals(goals)
    state = FeatureState(derive_confederations(matches))

    rows = []
    for m in matches.itertuples(index=False):
        ctx = MatchContext(
            date=m.match_date,
            tournament=m.tournament,
            neutral=bool(m.neutral),
            city=m.city,
            country=m.country,
        )
        rows.append(state.features(m.home_team, m.away_team, ctx))
        if m.is_played:
            state.update(
                m.home_team,
                m.away_team,
                ctx,
                int(m.home_score),
                int(m.away_score),
                goal_index.get((m.match_date, m.home_team, m.away_team), []),
            )

    features = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    return pd.concat([matches[META_COLUMNS], features], axis=1), state
