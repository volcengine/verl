# Unit tests for numeric canonicalization helpers:
# - legal integer strings      -> normalized form
# - illegal / non-integer str. -> unchanged
# - end-to-end `is_equiv` on boxed answers

import pytest
from verl.utils.reward_score.math import _canonical_int_if_safe, is_equiv, remove_boxed


# ----- legal integers -------------------------------------------------
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("033", "33"),
        ("+033", "33"),
        ("-0",  "0"),
        ("000", "0"),
    ],
)
def test_canonical_int_legal(raw, expected):
    """Legal integer strings should be canonicalised."""
    assert _canonical_int_if_safe(raw) == expected


# ----- illegal / non-integer ------------------------------------------
@pytest.mark.parametrize(
    "illegal",
    [
        "3.3",      # decimal
        "foo",      # letters
        "--33",     # malformed sign
        "",         # empty
        "+",        # sign only
        "33a",      # alphanumeric
    ],
)
def test_canonical_int_illegal(illegal):
    """Illegal strings must stay untouched."""
    assert _canonical_int_if_safe(illegal) == illegal


# ----- end-to-end boxed answers ---------------------------------------
@pytest.mark.parametrize(
    "gt, pred",
    [
        ("\\boxed{033}", "33"),
        ("\\boxed{+033}", "33"),
        ("\\boxed{-0}",  "0"),
        ("\\boxed{000}", "0"),
    ],
)
def test_is_equiv_boxed(gt, pred):
    """`is_equiv` should treat canonicalised values as equal."""
    gt_val = remove_boxed(gt)
    assert is_equiv(gt_val, pred) is True
