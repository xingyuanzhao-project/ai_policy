"""Tests for the filter normalization layer.

The normalizer is the server-side defense layer that turns whatever the LLM
emits (``"TX"``, ``"texas"``, ``"Texas"``, ``"developer"``) into the exact
canonical form stored in the QA corpus. The retrievers and metadata masks
downstream do plain exact-equality matching (``np.isin`` / set membership),
so every robustness guarantee the system advertises (USPS <-> full-name
aliasing, case-folding, topic fuzzy match) lives in this module.

These tests intentionally cover both the canonical-as-full-name case
(``canonical_states = ["California", "Texas"]``) and the canonical-as-abbrev
case (``canonical_states = ["CA", "TX"]``) because the test corpus and the
production corpus currently disagree on which form they store, and the
normalizer is expected to resolve both in either direction.
"""

from __future__ import annotations

import unittest

from src.qa.filter_normalizers import (
    US_STATE_ABBREV_TO_NAME,
    US_STATE_NAME_TO_ABBREV,
    normalize_categorical,
    normalize_categoricals,
    normalize_state,
    normalize_states,
    normalize_topic,
    normalize_topics,
)


class StateAbbreviationMapTests(unittest.TestCase):
    """Sanity-check the USPS <-> full-name tables."""

    def test_abbrev_to_name_covers_all_fifty_states(self) -> None:
        """Verify the map covers every US state abbreviation."""

        fifty_states = {
            "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
            "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
            "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
            "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
            "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        }

        for abbrev in fifty_states:
            self.assertIn(abbrev, US_STATE_ABBREV_TO_NAME)

    def test_name_to_abbrev_round_trip(self) -> None:
        """Verify every abbrev in the forward map has a reverse entry."""

        for abbrev, full_name in US_STATE_ABBREV_TO_NAME.items():
            self.assertEqual(US_STATE_NAME_TO_ABBREV[full_name.lower()], abbrev)


class NormalizeStateTests(unittest.TestCase):
    """Exercise every resolution branch documented in ``normalize_state``."""

    _CANONICAL_FULL = ("California", "Texas", "New York")
    _CANONICAL_ABBREV = ("CA", "TX", "NY")

    def test_exact_match_returns_canonical_casing(self) -> None:
        """Verify an already-canonical full name resolves unchanged."""

        self.assertEqual(
            normalize_state("California", self._CANONICAL_FULL),
            "California",
        )

    def test_case_fold_full_name(self) -> None:
        """Verify lowercase full names resolve to canonical casing."""

        self.assertEqual(
            normalize_state("california", self._CANONICAL_FULL),
            "California",
        )
        self.assertEqual(
            normalize_state("CALIFORNIA", self._CANONICAL_FULL),
            "California",
        )
        self.assertEqual(
            normalize_state("  Texas  ", self._CANONICAL_FULL),
            "Texas",
        )

    def test_usps_abbrev_maps_to_canonical_full_name(self) -> None:
        """Verify USPS codes resolve when the corpus stores full names."""

        self.assertEqual(
            normalize_state("TX", self._CANONICAL_FULL),
            "Texas",
        )
        self.assertEqual(
            normalize_state("tx", self._CANONICAL_FULL),
            "Texas",
        )
        self.assertEqual(
            normalize_state("Ca", self._CANONICAL_FULL),
            "California",
        )

    def test_full_name_maps_to_canonical_abbrev(self) -> None:
        """Verify full names resolve when the corpus stores USPS codes."""

        self.assertEqual(
            normalize_state("California", self._CANONICAL_ABBREV),
            "CA",
        )
        self.assertEqual(
            normalize_state("texas", self._CANONICAL_ABBREV),
            "TX",
        )
        self.assertEqual(
            normalize_state("New York", self._CANONICAL_ABBREV),
            "NY",
        )

    def test_abbrev_passes_through_when_corpus_stores_abbrev(self) -> None:
        """Verify already-canonical USPS codes resolve unchanged."""

        self.assertEqual(
            normalize_state("CA", self._CANONICAL_ABBREV),
            "CA",
        )

    def test_unknown_value_returns_none(self) -> None:
        """Verify junk or non-US state values return ``None``."""

        self.assertIsNone(normalize_state("Ontario", self._CANONICAL_FULL))
        self.assertIsNone(normalize_state("ZZ", self._CANONICAL_FULL))
        self.assertIsNone(normalize_state("", self._CANONICAL_FULL))
        self.assertIsNone(normalize_state(None, self._CANONICAL_FULL))
        self.assertIsNone(normalize_state(42, self._CANONICAL_FULL))

    def test_empty_canonical_returns_stripped_input(self) -> None:
        """Verify empty vocabulary degrades gracefully to stripped input.

        This is the cold-boot / test-harness path where no corpus has been
        loaded yet. Returning ``None`` here would leave callers with no
        filter at all; returning the stripped input preserves the user's
        intent even though it may not match anything downstream.
        """

        self.assertEqual(normalize_state("  ca  ", ()), "ca")

    def test_state_not_in_canonical_but_valid_usps_returns_none(self) -> None:
        """Verify a valid USPS code is still dropped if the corpus lacks it.

        Example: corpus only stores California bills, user queries ``"WY"``.
        The normalizer must NOT invent a Wyoming filter against a
        California-only corpus.
        """

        self.assertIsNone(normalize_state("WY", self._CANONICAL_FULL))


class NormalizeStatesTests(unittest.TestCase):
    """Verify the iterable wrapper around ``normalize_state``."""

    _CANONICAL = ("California", "Texas", "New York")

    def test_preserves_first_seen_order_and_dedupes(self) -> None:
        """Verify duplicates collapse and order is first-seen."""

        result = normalize_states(
            ["TX", "california", "Texas", "CA"], self._CANONICAL
        )
        self.assertEqual(result, ["Texas", "California"])

    def test_drops_unknown_values_silently(self) -> None:
        """Verify unknowns are omitted rather than raising."""

        result = normalize_states(["CA", "Ontario", "TX"], self._CANONICAL)
        self.assertEqual(result, ["California", "Texas"])

    def test_empty_input_returns_empty_list(self) -> None:
        """Verify ``[]`` input yields ``[]`` output."""

        self.assertEqual(normalize_states([], self._CANONICAL), [])


class NormalizeCategoricalTests(unittest.TestCase):
    """Exercise the case-fold-exact-match categorical helper."""

    _CANONICAL = ("Enacted", "Pending", "Introduced", "Vetoed", "Failed")

    def test_exact_match(self) -> None:
        """Verify canonical casing passes through unchanged."""

        self.assertEqual(normalize_categorical("Enacted", self._CANONICAL), "Enacted")

    def test_case_fold_match(self) -> None:
        """Verify varied casing resolves to canonical form."""

        self.assertEqual(normalize_categorical("enacted", self._CANONICAL), "Enacted")
        self.assertEqual(normalize_categorical("ENACTED", self._CANONICAL), "Enacted")
        self.assertEqual(normalize_categorical("  pending  ", self._CANONICAL), "Pending")

    def test_unknown_returns_none(self) -> None:
        """Verify unknown values resolve to ``None``."""

        self.assertIsNone(normalize_categorical("Withdrawn", self._CANONICAL))
        self.assertIsNone(normalize_categorical("", self._CANONICAL))
        self.assertIsNone(normalize_categorical(None, self._CANONICAL))


class NormalizeCategoricalsTests(unittest.TestCase):
    """Verify the iterable wrapper for categorical normalization."""

    def test_preserves_order_and_dedupes(self) -> None:
        """Verify duplicates collapse and order is first-seen."""

        result = normalize_categoricals(
            ["ENACTED", "pending", "Enacted"],
            ("Enacted", "Pending", "Introduced"),
        )
        self.assertEqual(result, ["Enacted", "Pending"])


class NormalizeTopicTests(unittest.TestCase):
    """Exercise exact + fuzzy resolution for topic tags.

    The topic vocabulary is more volatile than state/status (new topic
    labels appear whenever the corpus grows) so the normalizer layers a
    ``difflib`` fuzzy fallback on top of case-folding. These tests pin the
    exact behavior so a future cutoff change is visible in the diff.
    """

    _CANONICAL = (
        "Private Sector Use",
        "Government Use",
        "Employment",
        "Generative AI Disclosure",
    )

    def test_exact_match(self) -> None:
        """Verify canonical casing passes through unchanged."""

        self.assertEqual(
            normalize_topic("Private Sector Use", self._CANONICAL),
            "Private Sector Use",
        )

    def test_case_fold_match(self) -> None:
        """Verify varied casing resolves exactly before fuzzy fallback."""

        self.assertEqual(
            normalize_topic("private sector use", self._CANONICAL),
            "Private Sector Use",
        )
        self.assertEqual(
            normalize_topic("GOVERNMENT USE", self._CANONICAL),
            "Government Use",
        )

    def test_fuzzy_match_on_small_typo(self) -> None:
        """Verify a single-character typo still resolves via difflib."""

        resolved = normalize_topic("goverment use", self._CANONICAL, fuzzy_cutoff=0.7)
        self.assertEqual(resolved, "Government Use")

    def test_fuzzy_respects_cutoff(self) -> None:
        """Verify a far-off input is rejected at the default cutoff."""

        self.assertIsNone(normalize_topic("weather", self._CANONICAL))

    def test_lower_cutoff_accepts_looser_match(self) -> None:
        """Verify lowering the cutoff is the caller-facing permissiveness knob."""

        self.assertEqual(
            normalize_topic("employ", self._CANONICAL, fuzzy_cutoff=0.5),
            "Employment",
        )


class NormalizeTopicsTests(unittest.TestCase):
    """Verify the iterable wrapper for topic normalization."""

    _CANONICAL = ("Private Sector Use", "Government Use", "Employment")

    def test_mixed_inputs_resolve_together(self) -> None:
        """Verify a mix of exact, case-folded, and unknown inputs behaves right."""

        result = normalize_topics(
            ["private sector use", "GOVERNMENT USE", "unknown topic"],
            self._CANONICAL,
        )
        self.assertEqual(result, ["Private Sector Use", "Government Use"])


if __name__ == "__main__":
    unittest.main()
