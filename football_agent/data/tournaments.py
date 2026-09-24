"""Tournament tier classification — the single source of truth for Elo K-factors.

This is a corrected reimplementation of `FootballElo.classify_tournament` in
`reference/enhanced_features_upstream.py`, which has two defects:

1. **Ordering.** Upstream scans its tournament-name dictionary before testing for
   "qualification"/"qualifier". Because the test is a substring match,
   "FIFA World Cup qualification" matches the key "FIFA World Cup" and is graded
   as a World Cup final (K=60) rather than a qualifier (K=40). The same happens
   to Euro, AFCON, Asian Cup and Oceania qualifiers. That is 14,782 of 49,287
   matches — 30% of the dataset — carrying an inflated K-factor, so qualifying
   results move Elo ratings 25-50% more than they should.

2. **Accents.** The dictionary key is "Copa America" but the data says
   "Copa América". The substring test fails, so all 877 Copa América matches
   fall through to the default tier instead of being graded as a continental
   championship.

Both are fixed here: specificity-ordered tests, over accent-folded names.
"""

from __future__ import annotations

import unicodedata

#: Elo K-factor per tier. Higher K = a result moves the rating further.
K_FACTORS: dict[str, int] = {
    "world_cup": 60,
    "continental": 50,
    "qualifier": 40,
    "friendly": 20,
}

#: Substring -> tier, tested against accent-folded lowercase names.
#: Only reached after the qualifier and friendly tests below have been ruled out.
_CONTINENTAL = (
    "copa america",
    "uefa euro",
    "african cup of nations",
    "afc asian cup",
    "concacaf gold cup",
    "oceania nations cup",
    "confederations cup",
)

_WORLD_CUP = "fifa world cup"

#: Continental in name, but a season-long competitive league rather than a
#: finals tournament — upstream grades it as a qualifier and we keep that.
_NATIONS_LEAGUE = "uefa nations league"


def _fold(name: str) -> str:
    """Lowercase and strip diacritics: 'Copa América' -> 'copa america'."""
    decomposed = unicodedata.normalize("NFKD", name)
    return "".join(c for c in decomposed if not unicodedata.combining(c)).lower()


def classify(tournament: str) -> str:
    """Return the tier of a tournament: world_cup, continental, qualifier or friendly.

    Tests run most-specific first, so a qualifier is never mistaken for the
    finals it qualifies for.
    """
    name = _fold(tournament)

    if "friendly" in name:
        return "friendly"
    if "qualification" in name or "qualifier" in name:
        return "qualifier"
    if _NATIONS_LEAGUE in name:
        return "qualifier"
    if _WORLD_CUP in name:
        return "world_cup"
    if any(c in name for c in _CONTINENTAL):
        return "continental"

    # Regional cups, minor tournaments and one-off competitions: competitive,
    # but below finals intensity.
    return "qualifier"


def k_factor(tournament: str) -> int:
    """Elo K-factor for a tournament."""
    return K_FACTORS[classify(tournament)]
