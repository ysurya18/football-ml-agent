"""Venue and geography: altitude and confederation.

Two lookups replace upstream's hand-written tables in
`reference/enhanced_features_upstream.py`:

- **Altitude** keeps upstream's 24 cities, and matches accent-folded names:
  upstream's key 'bogota' never matched the data's "Bogotá" (nor 'sao paulo'
  "São Paulo"), the same defect as Copa América in the tournament tiers. It
  also adds the high-altitude cities that actually host matches in the data.
- **Confederation** is derived from the data instead of a 81-entry dictionary.
  Upstream's map left two-thirds of post-1990 matches with at least one team
  unmapped, and scored two unmapped teams as the *same* confederation.
"""

from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd

from football_agent.data.tournaments import _fold

#: Metres above sea level, keyed by accent-folded lowercase city name.
#: Approximate values from general reference; what matters most is which side
#: of HIGH_ALTITUDE a city falls. Names that denote more than one venue in the
#: data (León is in Mexico and Spain) are left out and take the default.
CITY_ALTITUDES: dict[str, int] = {
    # From upstream
    "mexico city": 2240,
    "bogota": 2640,
    "quito": 2850,
    "la paz": 3640,
    "johannesburg": 1753,
    "addis ababa": 2355,
    "nairobi": 1795,
    "denver": 1609,
    "madrid": 667,
    "sao paulo": 760,
    "guadalajara": 1566,
    "monterrey": 540,
    "atlanta": 320,
    "dallas": 131,
    "houston": 15,
    "kansas city": 247,
    "los angeles": 30,
    "miami": 2,
    "new york": 3,
    "philadelphia": 12,
    "san francisco": 16,
    "seattle": 54,
    "toronto": 76,
    "vancouver": 0,
    # Added: cities above 1,000 m that host internationals in the data
    "el alto": 4150,
    "oruro": 3735,
    "cusco": 3400,
    "sucre": 2810,
    "toluca": 2667,
    "ambato": 2580,
    "cochabamba": 2560,
    "cuenca": 2560,
    "arequipa": 2335,
    "thimphu": 2330,
    "asmara": 2325,
    "sana'a": 2250,
    "manizales": 2150,
    "puebla": 2135,
    "queretaro": 1820,
    "kabul": 1790,
    "windhoek": 1655,
    "albuquerque": 1620,
    "maseru": 1600,
    "kigali": 1570,
    "guatemala city": 1500,
    "harare": 1490,
    "kathmandu": 1400,
    "bloemfontein": 1395,
    "ulaanbaatar": 1350,
    "pretoria": 1340,
    "polokwane": 1310,
    "salt lake city": 1290,
    "lusaka": 1280,
    "mbabane": 1240,
    "kampala": 1190,
    "tehran": 1190,
    "rustenburg": 1170,
    "lilongwe": 1050,
    "gaborone": 1010,
}

DEFAULT_ALTITUDE = 100
HIGH_ALTITUDE = 1500

#: Competitions each confederation runs, as accent-folded names with any
#: " qualification" suffix removed. Copa América invites guests (Mexico, Japan,
#: Qatar, ...), which the majority vote in derive_confederations absorbs.
CONFEDERATION_COMPETITIONS: dict[str, str] = {
    "uefa euro": "UEFA",
    "uefa nations league": "UEFA",
    "copa america": "CONMEBOL",
    "gold cup": "CONCACAF",
    "concacaf championship": "CONCACAF",
    "concacaf nations league": "CONCACAF",
    "afc asian cup": "AFC",
    "afc challenge cup": "AFC",
    "african cup of nations": "CAF",
    "oceania nations cup": "OFC",
}

#: Upstream's prior on confederation strength. Hand-set, not learned.
CONFED_STRENGTH: dict[str, float] = {
    "UEFA": 1.0,
    "CONMEBOL": 0.95,
    "CONCACAF": 0.6,
    "AFC": 0.5,
    "CAF": 0.5,
    "OFC": 0.3,
}
UNKNOWN = "OTHER"
UNKNOWN_STRENGTH = 0.4


def _competition(tournament: str) -> str | None:
    name = _fold(tournament).removesuffix(" qualification")
    return CONFEDERATION_COMPETITIONS.get(name)


def derive_confederations(matches: pd.DataFrame) -> dict[str, str]:
    """Assign each team the confederation whose competitions it plays in most.

    Uses who took part, never who won, so it reveals nothing about results.
    It does use the whole history, so a team that switched confederation
    (Australia, OFC to AFC in 2006; Israel, AFC to UEFA) gets the one it has
    played most in throughout. Teams outside every confederation — CONIFA
    sides, island XIs — are absent and resolve to UNKNOWN.
    """
    votes: dict[str, Counter] = defaultdict(Counter)
    for tournament, home, away in matches[["tournament", "home_team", "away_team"]].itertuples(
        index=False
    ):
        confed = _competition(tournament)
        if confed is not None:
            votes[home][confed] += 1
            votes[away][confed] += 1
    return {team: c.most_common(1)[0][0] for team, c in votes.items()}


def altitude(city: str | None) -> int:
    if not city or pd.isna(city):
        return DEFAULT_ALTITUDE
    return CITY_ALTITUDES.get(_fold(city).strip(), DEFAULT_ALTITUDE)


def venue_features(
    home: str,
    away: str,
    city: str | None,
    confederations: dict[str, str],
    altitude_override: float | None = None,
) -> dict[str, float]:
    alt = altitude(city) if altitude_override is None else altitude_override
    h = confederations.get(home, UNKNOWN)
    a = confederations.get(away, UNKNOWN)
    # Two teams outside every confederation are not in the same one.
    same = h == a and h != UNKNOWN
    return {
        "altitude": alt,
        "is_high_altitude": int(alt > HIGH_ALTITUDE),
        "same_confederation": int(same),
        "confed_strength_diff": CONFED_STRENGTH.get(h, UNKNOWN_STRENGTH)
        - CONFED_STRENGTH.get(a, UNKNOWN_STRENGTH),
        "is_intercontinental": int(not same),
    }
