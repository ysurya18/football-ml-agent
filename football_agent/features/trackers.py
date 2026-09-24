"""Running per-team state, one tracker per feature family.

Every tracker has the same two-phase shape:

- a **read** method that returns features from the current state and never
  mutates it, and
- an **update** method that folds in one finished match.

The pipeline always reads before it updates, so a match's own result can never
reach its own feature row. Keeping the phases separate (upstream fused them in
`FootballElo.update`) is also what makes scenarios possible: a hypothetical
match is just a read with a different context.

Windows are bounded deques sized to the longest lookback each tracker needs, so
the state stays small and pickles cleanly for serving.

Feature semantics follow `reference/enhanced_features_upstream.py`, including
its defaults for teams with too little history. Deliberate departures are
called out where they occur.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import date

from football_agent.data.tournaments import K_FACTORS, classify


def points(gf: int, ga: int) -> float:
    """1 for a win, 0.5 for a draw, 0 for a loss."""
    return 1.0 if gf > ga else (0.5 if gf == ga else 0.0)


def _mean(xs) -> float:
    return sum(xs) / len(xs)


def _window(windows: dict[str, deque], team: str, size: int) -> deque:
    """The team's bounded history, created on first use.

    Plain dicts rather than defaultdict(lambda: ...), which cannot be pickled.
    """
    w = windows.get(team)
    if w is None:
        w = windows[team] = deque(maxlen=size)
    return w


# --- Elo -------------------------------------------------------------------------


class Elo:
    """Overall Elo plus a separate rating per tournament tier."""

    INITIAL = 1500.0
    HOME_ADVANTAGE = 100.0

    def __init__(self) -> None:
        self.ratings: dict[str, float] = {}
        self.tier_ratings: dict[str, dict[str, float]] = {tier: {} for tier in K_FACTORS}

    @staticmethod
    def expected(ra: float, rb: float, ha: float = 0.0) -> float:
        return 1.0 / (1.0 + 10.0 ** (-(ra - rb + ha) / 400.0))

    @staticmethod
    def goal_diff_multiplier(gd: int) -> float:
        gd = abs(gd)
        if gd <= 1:
            return 1.0
        if gd == 2:
            return 1.5
        if gd == 3:
            return 1.75
        return 1.75 + (gd - 3) / 8

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.INITIAL)

    def tier_rating(self, tier: str, team: str) -> float:
        return self.tier_ratings[tier].get(team, self.INITIAL)

    def features(self, home: str, away: str, tournament: str, neutral: bool) -> dict[str, float]:
        tier = classify(tournament)
        h, a = self.rating(home), self.rating(away)
        ht, at = self.tier_rating(tier, home), self.tier_rating(tier, away)
        return {
            "home_elo": h,
            "away_elo": a,
            "elo_diff": h - a,
            "elo_total": h + a,
            "home_tournament_elo": ht,
            "away_tournament_elo": at,
            "tournament_elo_diff": ht - at,
            "home_expected": self.expected(h, a, 0.0 if neutral else self.HOME_ADVANTAGE),
        }

    def update(
        self, home: str, away: str, hs: int, as_: int, tournament: str, neutral: bool
    ) -> None:
        tier = classify(tournament)
        k = K_FACTORS[tier]
        ha = 0.0 if neutral else self.HOME_ADVANTAGE
        g = self.goal_diff_multiplier(hs - as_)
        actual = points(hs, as_)

        for table, get in (
            (self.ratings, self.rating),
            (self.tier_ratings[tier], lambda t: self.tier_rating(tier, t)),
        ):
            rh, ra = get(home), get(away)
            exp = self.expected(rh, ra, ha)
            table[home] = rh + k * g * (actual - exp)
            table[away] = ra + k * g * ((1 - actual) - (1 - exp))


# --- Form, goals, rest -------------------------------------------------------------


@dataclass
class TeamForm:
    """Recent results, goal averages, rest days and experience for one team."""

    history: deque = field(default_factory=lambda: deque(maxlen=20))  # (gf, ga, pts)
    last_date: date | None = None
    n_matches: int = 0

    def _recent(self, n: int) -> list[tuple[int, int, float]]:
        return list(self.history)[-n:]

    def form(self, n: int) -> float:
        recent = self._recent(n)
        return _mean([r[2] for r in recent]) if len(recent) >= 3 else 0.5

    def weighted_form(self, n: int = 10, decay: float = 0.9) -> float:
        recent = self._recent(n)
        if len(recent) < 3:
            return 0.5
        # Newest match has weight 1, the one before decay, then decay^2, ...
        weights = [decay**i for i in range(len(recent) - 1, -1, -1)]
        return sum(w * r[2] for w, r in zip(weights, recent)) / sum(weights)

    def goals_scored(self, n: int) -> float:
        recent = self._recent(n)
        return _mean([r[0] for r in recent]) if len(recent) >= 3 else 1.5

    def goals_conceded(self, n: int) -> float:
        recent = self._recent(n)
        return _mean([r[1] for r in recent]) if len(recent) >= 3 else 1.5

    def goal_diff(self, n: int) -> float:
        recent = self._recent(n)
        return _mean([r[0] - r[1] for r in recent]) if len(recent) >= 3 else 0.0

    def days_rest(self, on: date) -> int:
        return 30 if self.last_date is None else (on - self.last_date).days

    def update(self, on: date, gf: int, ga: int) -> None:
        self.history.append((gf, ga, points(gf, ga)))
        self.last_date = on
        self.n_matches += 1


# --- Head to head ---------------------------------------------------------------------


@dataclass
class _Pair:
    total: int = 0
    wins: Counter = field(default_factory=Counter)
    goals: Counter = field(default_factory=Counter)


class HeadToHead:
    def __init__(self) -> None:
        self.pairs: dict[tuple[str, str], _Pair] = defaultdict(_Pair)

    @staticmethod
    def _key(a: str, b: str) -> tuple[str, str]:
        return (a, b) if a <= b else (b, a)

    def features(self, a: str, b: str) -> dict[str, float]:
        """From `a`'s point of view."""
        rec = self.pairs.get(self._key(a, b))
        if rec is None or rec.total == 0:
            return {"h2h_win_rate": 0.5, "h2h_matches": 0, "h2h_goal_diff": 0.0}
        return {
            "h2h_win_rate": rec.wins[a] / rec.total,
            "h2h_matches": rec.total,
            "h2h_goal_diff": (rec.goals[a] - rec.goals[b]) / rec.total,
        }

    def update(self, home: str, away: str, hs: int, as_: int) -> None:
        rec = self.pairs[self._key(home, away)]
        rec.total += 1
        rec.goals[home] += hs
        rec.goals[away] += as_
        if hs != as_:
            rec.wins[home if hs > as_ else away] += 1


# --- Goalscorer patterns ------------------------------------------------------------------


@dataclass(frozen=True)
class Goal:
    """One row of the goals table. `team` is the side the goal counts for —
    for an own goal that is the *opponent* of the player who scored it."""

    team: str
    scorer: str
    minute: int | None
    own_goal: bool
    penalty: bool


class Goalscorers:
    """Timing, depth and star-dependency over each team's last 50 recorded goals."""

    WINDOW = 50

    def __init__(self) -> None:
        self.goals: dict[str, deque[Goal]] = {}

    def features(self, team: str) -> dict[str, float]:
        goals = self.goals.get(team, ())
        if len(goals) < 5:
            return {
                "scoring_depth": 0.5,
                "star_dependency": 0.5,
                "penalty_ratio": 0.1,
                "late_goal_ratio": 0.2,
                "first_half_ratio": 0.5,
            }

        minutes = [g.minute for g in goals if g.minute is not None]
        scorers = [g.scorer for g in goals if not g.own_goal]

        return {
            # Distinct scorers per goal: higher = more distributed threat.
            "scoring_depth": len(set(scorers)) / len(scorers) if scorers else 0.5,
            # Top scorer's share: higher = more reliant on one player.
            "star_dependency": (
                Counter(scorers).most_common(1)[0][1] / len(scorers) if scorers else 0.5
            ),
            "penalty_ratio": sum(g.penalty for g in goals) / len(goals),
            "late_goal_ratio": sum(m >= 75 for m in minutes) / len(minutes) if minutes else 0.2,
            "first_half_ratio": sum(m <= 45 for m in minutes) / len(minutes) if minutes else 0.5,
        }

    def update(self, match_goals: list[Goal]) -> None:
        for g in match_goals:
            _window(self.goals, g.team, self.WINDOW).append(g)


# --- Momentum ---------------------------------------------------------------------


def conceded_first(team: str, gf: int, ga: int, match_goals: list[Goal]) -> bool | None:
    """Did `team` concede the opening goal? None when it cannot be known.

    Two departures from upstream's `determine_conceded_first`:

    - **Own goals.** `Goal.team` is already the side the goal counts for.
      Upstream inverted it again for own goals, so a match opened by an own
      goal was credited to the wrong side (332 matches).
    - **Missing data.** Only about a third of scoring matches have goal-by-goal
      rows. Upstream fell back to "conceded at all" for the rest, so
      `comeback_rate` meant different things depending on data coverage. We
      return None instead and the match is left out of the rate.
    """
    if ga == 0:
        return False
    if gf == 0:
        return True
    timed = [g for g in match_goals if g.minute is not None]
    if not timed:
        return None
    opener = min(timed, key=lambda g: g.minute)
    return opener.team != team


class Momentum:
    """Streaks, clean sheets, comebacks and result shape over the last 15 matches."""

    WINDOW = 15

    def __init__(self) -> None:
        # (pts, gf, ga, conceded_first)
        self.results: dict[str, deque] = {}

    def features(self, team: str) -> dict[str, float]:
        results = list(self.results.get(team, ()))
        if len(results) < 5:
            return {
                "current_streak": 0,
                "unbeaten_streak": 0,
                "clean_sheet_pct": 0.3,
                "comeback_rate": 0.2,
                "draw_tendency": 0.25,
                "blowout_win_pct": 0.1,
                "blowout_loss_pct": 0.1,
                "shutout_loss_pct": 0.1,
            }

        def run(pred) -> int:
            n = 0
            for r in reversed(results):
                if not pred(r):
                    break
                n += 1
            return n

        n = len(results)
        behind = [r for r in results if r[3] is True]
        return {
            "current_streak": run(lambda r: r[0] == 1.0),
            "unbeaten_streak": run(lambda r: r[0] >= 0.5),
            "clean_sheet_pct": sum(r[2] == 0 for r in results) / n,
            "comeback_rate": sum(r[0] == 1.0 for r in behind) / len(behind) if behind else 0.2,
            "draw_tendency": sum(r[0] == 0.5 for r in results) / n,
            "blowout_win_pct": sum(r[1] - r[2] >= 3 for r in results) / n,
            "blowout_loss_pct": sum(r[2] - r[1] >= 3 for r in results) / n,
            "shutout_loss_pct": sum(r[1] == 0 and r[2] > 0 for r in results) / n,
        }

    def update(self, team: str, gf: int, ga: int, conceded: bool | None) -> None:
        _window(self.results, team, self.WINDOW).append((points(gf, ga), gf, ga, conceded))


# --- Poisson expected goals ----------------------------------------------------------


def _poisson_pmf(lam: float, max_goals: int) -> list[float]:
    pmf = [math.exp(-lam)]
    for k in range(1, max_goals + 1):
        pmf.append(pmf[-1] * lam / k)
    return pmf


class PoissonGoals:
    """Each side's scoring rate as a Poisson process over its last 20 matches.

    Two departures from upstream, both about exactness rather than intent:

    - Upstream cached PMFs by lambda rounded to 0.1, but filled each bucket
      with the PMF of whichever lambda reached it *first*. A feature value then
      depended on the order matches were processed in, so a live prediction
      could differ from the identical training row. We compute exactly.
    - Upstream truncated the score grid at 7 goals, which drops up to 13% of
      the probability mass at lambda=5. We go to 15, where the loss is
      negligible.
    """

    WINDOW = 20
    MAX_GOALS = 15

    def __init__(self) -> None:
        self.scored: dict[str, deque[int]] = {}
        self.conceded: dict[str, deque[int]] = {}

    def features(self, home: str, away: str) -> dict[str, float]:
        hs, hc = list(self.scored.get(home, ())), list(self.conceded.get(home, ()))
        as_, ac = list(self.scored.get(away, ())), list(self.conceded.get(away, ()))

        if len(hs) < 5 or len(as_) < 5:
            return {
                "home_lambda": 1.5,
                "away_lambda": 1.2,
                "home_poisson_win": 0.4,
                "home_poisson_draw": 0.25,
                "home_scoring_variance": 1.0,
                "away_scoring_variance": 1.0,
                "home_overperformance": 0.0,
                "away_overperformance": 0.0,
            }

        # Attack of one side against the defence of the other.
        home_lambda = min(max((_mean(hs) + _mean(ac)) / 2, 0.3), 5.0)
        away_lambda = min(max((_mean(as_) + _mean(hc)) / 2, 0.3), 5.0)

        h_pmf = _poisson_pmf(home_lambda, self.MAX_GOALS)
        a_pmf = _poisson_pmf(away_lambda, self.MAX_GOALS)
        win = sum(h_pmf[i] * a_pmf[j] for i in range(len(h_pmf)) for j in range(i))
        draw = sum(p * q for p, q in zip(h_pmf, a_pmf))

        def var(xs: list[int]) -> float:
            m = _mean(xs)
            return sum((x - m) ** 2 for x in xs) / len(xs)

        # As upstream: the team's recent win rate against *all* opponents, minus
        # the Poisson win probability for *this* fixture.
        h_wins = _mean([s > c for s, c in zip(hs, hc)])
        a_wins = _mean([s > c for s, c in zip(as_, ac)])

        return {
            "home_lambda": home_lambda,
            "away_lambda": away_lambda,
            "home_poisson_win": win,
            "home_poisson_draw": draw,
            "home_scoring_variance": var(hs),
            "away_scoring_variance": var(as_),
            "home_overperformance": h_wins - win,
            "away_overperformance": a_wins - (1 - win - draw),
        }

    def update(self, team: str, gf: int, ga: int) -> None:
        _window(self.scored, team, self.WINDOW).append(gf)
        _window(self.conceded, team, self.WINDOW).append(ga)


# --- Tournament context ------------------------------------------------------------


def stage(tournament: str) -> str:
    """friendly, qualifying, wc_finals, continental_finals or other_competitive.

    Built on the tier classifier rather than upstream's own substring list,
    which graded anything containing "copa" or "euro" as continental finals —
    Copa Newton, Copa Lipton, the Central European International Cup and ~20
    other minor cups, about 500 matches.
    """
    tier = classify(tournament)
    if tier == "friendly":
        return "friendly"
    if "qualif" in tournament.lower():
        return "qualifying"
    if tier == "world_cup":
        return "wc_finals"
    if tier == "continental":
        return "continental_finals"
    return "other_competitive"


class TournamentContext:
    """Form split by competition type, and World Cup finals experience."""

    WINDOW = 20

    def __init__(self) -> None:
        self.results: dict[str, dict[str, deque[float]]] = defaultdict(dict)
        self.wc_matches: Counter = Counter()

    def features(self, team: str) -> dict[str, float]:
        ctx = self.results.get(team, {})

        def form(key: str) -> float:
            r = ctx.get(key, ())
            return _mean(r) if len(r) >= 3 else 0.5

        competitive = (
            0.4 * form("wc_finals") + 0.3 * form("continental_finals") + 0.3 * form("qualifying")
        )
        return {
            "wc_form": form("wc_finals"),
            "competitive_form": competitive,
            # How much better (or worse) the team is when it matters.
            "big_game_factor": competitive - form("friendly"),
            "wc_experience": self.wc_matches[team],
        }

    def update(self, team: str, tournament: str, gf: int, ga: int) -> None:
        s = stage(tournament)
        self.results[team].setdefault(s, deque(maxlen=self.WINDOW)).append(points(gf, ga))
        if s == "wc_finals":
            self.wc_matches[team] += 1
