"""
Validation of the LM-CC implementation against Paper Proposition B.1 (Appendix B.2).

The paper proves theoretically:
  - flat sequential conditionals:  LM-CC = Theta(n)
  - deeply nested conditionals:    LM-CC = Theta(n^2)

at identical cyclomatic complexity (CC = n+1) and comparable length.

These tests verify that the implementation reproduces this behavior.
They validate the structural correctness of the LM-CC computation,
independently of the evaluation model.
"""

import numpy as np
import pytest

from src.lm_cc import compute_lm_cc

def make_flat(n: int) -> str:
    """n sequential, independent conditionals (all at level 1).

    Matches c_flat_n from Proposition B.1:
        if cond1: A1 else: B1
        if cond2: A2 else: B2
        ...
    CC = n + 1, all if-blocks at the same nesting level.
    """
    lines = []
    for i in range(n):
        lines.append(f"if cond{i} > 0:")
        lines.append(f"    a{i} = {i}")
        lines.append(f"else:")
        lines.append(f"    b{i} = {i}")
    return "\n".join(lines)


def make_nested(n: int) -> str:
    """n nested conditionals (levels 1 through n).

    Matches c_nested_n from Proposition B.1:
        if cond1:
            if cond2:
                ...
                    if condn: An else: Bn
                ...
            else: B2
        else: B1
    CC = n + 1, but nesting depth is n.
    """
    lines = []
    for i in range(n):
        indent = "    " * i
        lines.append(f"{indent}if cond{i} > 0:")
    # innermost body
    lines.append("    " * n + "x = 1")
    # else branches from inner to outer
    for i in range(n - 1, -1, -1):
        indent = "    " * i
        lines.append(f"{indent}else:")
        lines.append(f"{indent}    y{i} = {i}")
    return "\n".join(lines)

def lm_cc_of(code: str) -> float:
    """Extract the final LM-CC score."""
    return compute_lm_cc(code)["lm_cc_score"]

class TestPropositionB1:
    """Tests according to Paper Appendix B.2, Proposition B.1."""

    def test_nested_exceeds_flat(self):
        """For equal n, nested must have a higher LM-CC than flat."""
        for n in [3, 5, 7]:
            flat = lm_cc_of(make_flat(n))
            nested = lm_cc_of(make_nested(n))
            assert nested > flat, (
                f"n={n}: nested ({nested:.1f}) should be > flat ({flat:.1f})"
            )

    def test_nested_grows_faster(self):
        """The nested/flat ratio must grow with n (Theta(n^2) vs Theta(n))."""
        ns = [2, 4, 6, 8]
        ratios = []
        for n in ns:
            flat = lm_cc_of(make_flat(n))
            nested = lm_cc_of(make_nested(n))
            ratios.append(nested / flat)

        # The ratio should increase with n
        assert ratios[-1] > ratios[0], (
            f"nested/flat ratio should grow with n, but was {ratios}"
        )

    def test_flat_grows_linearly(self):
        """Flat LM-CC should grow approximately linearly with n."""
        ns = np.array([2, 4, 6, 8, 10])
        values = np.array([lm_cc_of(make_flat(n)) for n in ns])

        # Linear regression: R^2 should be high for a linear fit
        coeffs = np.polyfit(ns, values, 1)
        predicted = np.polyval(coeffs, ns)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - values.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        assert r_squared > 0.9, (
            f"Flat should grow linearly (R^2>0.9), but R^2={r_squared:.3f}, "
            f"values={values}"
        )

    def test_nested_grows_superlinearly(self):
        """Nested LM-CC should grow faster than linear (quadratic component)."""
        ns = np.array([2, 4, 6, 8, 10])
        values = np.array([lm_cc_of(make_nested(n)) for n in ns])

        # A quadratic fit should capture a positive leading coefficient
        quad_coeffs = np.polyfit(ns, values, 2)

        assert quad_coeffs[0] > 0, (
            f"Nested should have a positive quadratic term, "
            f"but was {quad_coeffs[0]:.3f}"
        )


class TestCompositionalLevel:
    """Tests that the compositional level reflects nesting depth."""

    def test_max_comp_tracks_depth(self):
        """max_comp should grow with nesting depth."""
        prev_max_comp = 0
        for depth in range(1, 6):
            code = make_nested(depth)
            max_comp = compute_lm_cc(code)["lm_cc_max_comp"]
            assert max_comp >= prev_max_comp, (
                f"depth {depth}: max_comp ({max_comp}) should be >= "
                f"previous ({prev_max_comp})"
            )
            prev_max_comp = max_comp

    def test_sequential_vs_nested_max_comp(self):
        """For equal if-count: nested has higher max_comp than sequential."""
        n = 4
        flat_max_comp = compute_lm_cc(make_flat(n))["lm_cc_max_comp"]
        nested_max_comp = compute_lm_cc(make_nested(n))["lm_cc_max_comp"]
        assert nested_max_comp > flat_max_comp, (
            f"nested max_comp ({nested_max_comp}) should be > "
            f"flat max_comp ({flat_max_comp})"
        )

    def test_total_comp_quadratic_nested(self):
        """TotalCompLevel (purely structural) should grow quadratically for nested.

        This is the cleanest test since total_comp does not depend on
        entropy and directly reflects the Theta(n^2) claim in Proposition B.1.
        """
        ns = np.array([2, 4, 6, 8, 10])
        values = np.array([
            compute_lm_cc(make_nested(n))["lm_cc_total_comp"]
            for n in ns
        ])
        quad_coeffs = np.polyfit(ns, values, 2)
        assert quad_coeffs[0] > 0, (
            f"Nested total_comp should have positive quadratic term, "
            f"but was {quad_coeffs[0]:.3f}, values={values}"
        )

    def test_total_comp_linear_flat(self):
        """TotalCompLevel should grow linearly for flat code (Theta(n))."""
        ns = np.array([2, 4, 6, 8, 10])
        values = np.array([
            compute_lm_cc(make_flat(n))["lm_cc_total_comp"]
            for n in ns
        ])
        coeffs = np.polyfit(ns, values, 1)
        predicted = np.polyval(coeffs, ns)
        ss_res = np.sum((values - predicted) ** 2)
        ss_tot = np.sum((values - values.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        assert r_squared > 0.9, (
            f"Flat total_comp should grow linearly (R^2>0.9), "
            f"but R^2={r_squared:.3f}, values={values}"
        )


class TestSanity:
    """Basic sanity checks."""

    def test_trivial_code_low_score(self):
        """A trivial single-line assignment should have minimal LM-CC."""
        score = lm_cc_of("x = 1")
        assert score >= 0
        # Should be lower than any nested code
        assert score < lm_cc_of(make_nested(3))

    def test_minimal_code_handled(self):
        """Minimal code does not crash."""
        result = compute_lm_cc("pass")
        assert "lm_cc_score" in result

    def test_deterministic(self):
        """Same code -> same score (reproducibility)."""
        code = make_nested(4)
        score1 = lm_cc_of(code)
        score2 = lm_cc_of(code)
        assert score1 == pytest.approx(score2)

class TestEdgeCases:
    """Tests for constructs that appear in real SWE-bench/HumanEval code."""

    def test_for_builds_hierarchy(self):
        flat = compute_lm_cc("for i in r:\n    x = i")
        nested = compute_lm_cc("for i in r:\n    for j in s:\n        x = i")
        assert nested["lm_cc_max_comp"] > flat["lm_cc_max_comp"]

    def test_while_builds_hierarchy(self):
        nested = compute_lm_cc("while a:\n    while b:\n        c = 1")
        assert nested["lm_cc_max_comp"] >= 2

    def test_try_except_builds_hierarchy(self):
        code = "try:\n    x = f()\nexcept ValueError:\n    x = 0"
        result = compute_lm_cc(code)
        assert result["lm_cc_max_comp"] >= 1

    def test_mixed_nesting_depth(self):
        """Mixed nesting: def→for→try→if should reach depth ~4."""
        code = (
            "def process(items):\n"
            "    for item in items:\n"
            "        try:\n"
            "            if item.valid:\n"
            "                x = item.value\n"
            "        except AttributeError:\n"
            "            continue"
        )
        result = compute_lm_cc(code)
        assert result["lm_cc_max_comp"] >= 3, (
            f"Mixed nesting should reach depth >=3, got {result['lm_cc_max_comp']}"
        )

    def test_multiple_functions_not_nested(self):
        """Two top-level functions should NOT be nested into each other."""
        code = (
            "def a():\n"
            "    if x:\n"
            "        return 1\n"
            "def b():\n"
            "    if y:\n"
            "        return 2"
        )
        result = compute_lm_cc(code)
        # max_comp should be ~2 (def→if), not 4 (as if nested)
        assert result["lm_cc_max_comp"] <= 3

    def test_unparseable_no_crash(self):
        result = compute_lm_cc("print 'python2 syntax'\nfor x in")
        assert "lm_cc_score" in result

    def test_unusual_indentation(self):
        """Unusual indentation (8 spaces) should be handled consistently."""
        code = "if x:\n        y = 1"  # 8-space indent
        result = compute_lm_cc(code)
        assert "lm_cc_score" in result