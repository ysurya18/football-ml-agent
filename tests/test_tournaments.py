"""Tests for tournament tier classification.

The first two classes are regression tests for defects in the upstream
implementation this module replaces (see football_agent/data/tournaments.py).
"""

import pytest

from football_agent.data.tournaments import K_FACTORS, classify, k_factor


class TestQualifiersAreNotFinals:
    """Upstream substring-matched the dictionary before testing for qualifiers,
    so every 'X qualification' was graded as the finals of X."""

    @pytest.mark.parametrize(
        "tournament",
        [
            "FIFA World Cup qualification",
            "UEFA Euro qualification",
            "African Cup of Nations qualification",
            "AFC Asian Cup qualification",
            "Oceania Nations Cup qualification",
            "CONCACAF Gold Cup qualification",
        ],
    )
    def test_qualifier_beats_the_competition_it_qualifies_for(self, tournament):
        assert classify(tournament) == "qualifier"
        assert k_factor(tournament) == 40

    def test_world_cup_finals_still_classify_as_finals(self):
        assert classify("FIFA World Cup") == "world_cup"
        assert k_factor("FIFA World Cup") == 60


class TestAccentedNames:
    """Upstream's key was 'Copa America'; the data says 'Copa América', so the
    substring test failed and 877 matches fell through to the default tier."""

    def test_copa_america_with_accent(self):
        assert classify("Copa América") == "continental"
        assert k_factor("Copa América") == 50

    def test_copa_america_without_accent(self):
        assert classify("Copa America") == "continental"

    def test_accented_qualifier_still_a_qualifier(self):
        assert classify("Copa América qualification") == "qualifier"


class TestTiers:
    @pytest.mark.parametrize(
        "tournament,tier",
        [
            ("Friendly", "friendly"),
            ("UEFA Euro", "continental"),
            ("African Cup of Nations", "continental"),
            ("AFC Asian Cup", "continental"),
            ("CONCACAF Gold Cup", "continental"),
            ("Confederations Cup", "continental"),
            ("UEFA Nations League", "qualifier"),
            ("CECAFA Cup", "qualifier"),
        ],
    )
    def test_known_tournaments(self, tournament, tier):
        assert classify(tournament) == tier

    def test_unknown_tournament_defaults_to_competitive(self):
        assert classify("Some Regional Invitational 1954") == "qualifier"

    def test_case_insensitive(self):
        assert classify("FIFA WORLD CUP") == classify("fifa world cup") == "world_cup"

    def test_every_tier_has_a_k_factor(self):
        tiers = {
            classify(t) for t in ["Friendly", "FIFA World Cup", "UEFA Euro", "X qualification"]
        }
        assert tiers <= set(K_FACTORS)

    def test_k_factors_are_ordered_by_stakes(self):
        assert (
            K_FACTORS["world_cup"]
            > K_FACTORS["continental"]
            > K_FACTORS["qualifier"]
            > K_FACTORS["friendly"]
        )
