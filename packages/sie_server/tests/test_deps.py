"""Tests for dependency resolution utilities."""

from sie_server.core.deps import _is_specifier_satisfiable, _merge_constraints


class TestMergeConstraints:
    """Tests for _merge_constraints."""

    def test_empty_constraints(self) -> None:
        assert _merge_constraints("pkg", []) == "pkg"

    def test_single_constraint(self) -> None:
        result = _merge_constraints("pkg", [(">=1.0", "source1")])
        assert result == "pkg>=1.0"

    def test_multiple_compatible_constraints(self) -> None:
        result = _merge_constraints(
            "pkg",
            [
                (">=1.0", "source1"),
                ("<2.0", "source2"),
            ],
        )
        assert result == "pkg>=1.0,<2.0"

    def test_duplicate_constraints_are_deduped(self) -> None:
        result = _merge_constraints(
            "pkg",
            [
                (">=1.0,<2.0", "source1"),
                (">=1.0,<2.0", "source2"),
                (">=1.0,<2.0", "source3"),
            ],
        )
        assert result == "pkg>=1.0,<2.0"

    def test_overlapping_specifiers_deduped(self) -> None:
        """Specifiers from different constraint strings are deduped individually."""
        result = _merge_constraints(
            "pkg",
            [
                (">=0.2,<1", "source1"),
                ("<1,>=0.2", "source2"),
            ],
        )
        assert result == "pkg>=0.2,<1"

    def test_torch_pin_deduped(self) -> None:
        """Multiple adapters declaring the same torch pin should deduplicate."""
        result = _merge_constraints(
            "torch",
            [
                (">=2.9,<2.10", "adapter1"),
                (">=2.9,<2.10", "adapter2"),
                (">=2.9,<2.10", "adapter3"),
            ],
        )
        assert result == "torch>=2.9,<2.10"

    def test_torch_mixed_constraints_deduped(self) -> None:
        """Torch pin plus broader constraint should merge cleanly."""
        result = _merge_constraints(
            "torch",
            [
                ("<2.10,>=2.9", "adapter1"),
                (">=2.0", "adapter2"),
                ("<2.10,>=2.9", "adapter3"),
            ],
        )
        assert result == "torch<2.10,>=2.9,>=2.0"

    def test_conflicting_constraints(self) -> None:
        result = _merge_constraints(
            "pkg",
            [
                (">=5.0", "source1"),
                ("<4.0", "source2"),
            ],
        )
        assert result.startswith("CONFLICT:")

    def test_url_dependency(self) -> None:
        url = "flash-attn @ https://example.com/wheel.whl ; sys_platform == 'linux'"
        result = _merge_constraints("flash-attn", [(url, "source1")])
        assert result == url

    def test_duplicate_url_dependencies(self) -> None:
        url = "flash-attn @ https://example.com/wheel.whl"
        result = _merge_constraints(
            "flash-attn",
            [
                (url, "source1"),
                (url, "source2"),
            ],
        )
        assert result == url

    def test_conflicting_url_dependencies(self) -> None:
        result = _merge_constraints(
            "flash-attn",
            [
                ("flash-attn @ https://example.com/v1.whl", "source1"),
                ("flash-attn @ https://example.com/v2.whl", "source2"),
            ],
        )
        assert result.startswith("CONFLICT:")

    def test_empty_constraint_strings(self) -> None:
        result = _merge_constraints(
            "pkg",
            [
                ("", "source1"),
                ("", "source2"),
            ],
        )
        assert result == "pkg"


class TestIsSpecifierSatisfiable:
    """Tests for _is_specifier_satisfiable."""

    def test_simple_range(self) -> None:
        from packaging.specifiers import SpecifierSet

        assert _is_specifier_satisfiable(SpecifierSet(">=1.0,<2.0")) is True

    def test_impossible_range(self) -> None:
        from packaging.specifiers import SpecifierSet

        assert _is_specifier_satisfiable(SpecifierSet(">=5.0,<4.0")) is False

    def test_torch_range(self) -> None:
        """The torch pin >=2.9,<2.10 must be recognized as satisfiable."""
        from packaging.specifiers import SpecifierSet

        assert _is_specifier_satisfiable(SpecifierSet(">=2.9,<2.10")) is True

    def test_transformers_range(self) -> None:
        from packaging.specifiers import SpecifierSet

        assert _is_specifier_satisfiable(SpecifierSet(">=4.45,<5")) is True

    def test_exact_version(self) -> None:
        from packaging.specifiers import SpecifierSet

        assert _is_specifier_satisfiable(SpecifierSet("==2.9.1")) is True

    def test_single_lower_bound(self) -> None:
        from packaging.specifiers import SpecifierSet

        assert _is_specifier_satisfiable(SpecifierSet(">=0.1")) is True
