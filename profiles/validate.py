"""Profile validation CLI tool.

Run with: python -m profiles.validate

Validates all profiles for:
- Required fields
- Token count limits
- Keyword conflicts
- Import errors
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import Any


# ANSI colors
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def colorize(text: str, color: str) -> str:
    """Add ANSI color to text."""
    return f"{color}{text}{Colors.RESET}"


def estimate_tokens(text: str) -> int:
    """Rough token estimate (words * 1.3)."""
    return int(len(text.split()) * 1.3)


# Required fields for a valid profile
REQUIRED_FIELDS = [
    "PROFILE_NAME",
    "PROFILE_VERSION",
    "TECHNOLOGIES",
    "TRIGGER_KEYWORDS",
    "DEPENDENCIES",
]

REQUIRED_FUNCTIONS = [
    "should_apply",
    "get_guidance",
    "get_dependencies",
]

# Recommended fields
RECOMMENDED_FIELDS = [
    "DESCRIPTION",
    "AUTHOR",
    "OPTIONAL_DEPENDENCIES",
]

# Token limits
MAX_GUIDANCE_TOKENS = 2000
RECOMMENDED_GUIDANCE_TOKENS = 1500


def validate_profile(module: Any, verbose: bool = False) -> tuple[bool, list[str], list[str]]:
    """Validate a single profile module.

    Args:
        module: The profile module
        verbose: Print detailed info

    Returns:
        Tuple of (is_valid, errors, warnings)
    """
    errors = []
    warnings = []
    profile_name = getattr(module, "PROFILE_NAME", module.__name__)

    # Check required fields
    for field in REQUIRED_FIELDS:
        if not hasattr(module, field):
            errors.append(f"Missing required field: {field}")
        elif not getattr(module, field):
            errors.append(f"Empty required field: {field}")

    # Check required functions
    for func in REQUIRED_FUNCTIONS:
        if not hasattr(module, func):
            errors.append(f"Missing required function: {func}()")
        elif not callable(getattr(module, func)):
            errors.append(f"{func} is not callable")

    # Check recommended fields
    for field in RECOMMENDED_FIELDS:
        if not hasattr(module, field):
            warnings.append(f"Missing recommended field: {field}")

    # Validate guidance tokens
    if hasattr(module, "get_guidance"):
        try:
            guidance = module.get_guidance()
            tokens = estimate_tokens(guidance)

            if tokens > MAX_GUIDANCE_TOKENS:
                errors.append(
                    f"Guidance too long: ~{tokens} tokens (max {MAX_GUIDANCE_TOKENS})"
                )
            elif tokens > RECOMMENDED_GUIDANCE_TOKENS:
                warnings.append(
                    f"Guidance is long: ~{tokens} tokens (recommended <{RECOMMENDED_GUIDANCE_TOKENS})"
                )
            elif verbose:
                print(f"  Guidance tokens: ~{tokens}")

        except Exception as e:
            errors.append(f"get_guidance() failed: {e}")

    # Validate should_apply function signature
    if hasattr(module, "should_apply"):
        try:
            # Test with typical inputs
            result = module.should_apply(["test"], "test prompt")
            if not isinstance(result, bool):
                errors.append(f"should_apply() should return bool, got {type(result)}")
        except TypeError as e:
            errors.append(f"should_apply() has wrong signature: {e}")
        except Exception as e:
            errors.append(f"should_apply() failed: {e}")

    # Validate get_dependencies
    if hasattr(module, "get_dependencies"):
        try:
            deps = module.get_dependencies()
            if not isinstance(deps, list):
                errors.append(f"get_dependencies() should return list, got {type(deps)}")

            # Check with features
            deps_with_features = module.get_dependencies(["database", "auth"])
            if not isinstance(deps_with_features, list):
                errors.append("get_dependencies(features) should return list")

        except TypeError as e:
            errors.append(f"get_dependencies() has wrong signature: {e}")
        except Exception as e:
            errors.append(f"get_dependencies() failed: {e}")

    # Validate keywords don't have common false-positive issues
    keywords = getattr(module, "TRIGGER_KEYWORDS", [])
    short_keywords = [kw for kw in keywords if len(kw) <= 2]
    if short_keywords:
        warnings.append(f"Very short keywords may cause false positives: {short_keywords}")

    # Check for version format
    version = getattr(module, "PROFILE_VERSION", "")
    if version and not all(c.isdigit() or c == "." for c in version):
        warnings.append(f"Version '{version}' should be numeric (e.g., '1.0')")

    is_valid = len(errors) == 0
    return is_valid, errors, warnings


def discover_profiles() -> list[tuple[str, Any | None, str | None]]:
    """Discover all profile modules.

    Returns:
        List of (name, module, error) tuples
    """
    profiles_dir = Path(__file__).parent
    profiles = []

    for file in sorted(profiles_dir.glob("*.py")):
        if file.stem.startswith("_") or file.stem in ("base", "manager", "validate"):
            continue

        try:
            module = importlib.import_module(f"profiles.{file.stem}")
            if hasattr(module, "PROFILE_NAME"):
                profiles.append((file.stem, module, None))
            else:
                profiles.append((file.stem, None, "Not a profile (no PROFILE_NAME)"))
        except ImportError as e:
            profiles.append((file.stem, None, f"Import error: {e}"))
        except Exception as e:
            profiles.append((file.stem, None, f"Error: {e}"))

    return profiles


def check_keyword_conflicts(profiles: list[tuple[str, Any | None, str | None]]) -> list[str]:
    """Check for keyword conflicts between profiles.

    Returns:
        List of conflict warnings
    """
    keyword_to_profiles: dict[str, list[str]] = {}

    for name, module, _ in profiles:
        if module is None:
            continue

        keywords = getattr(module, "TRIGGER_KEYWORDS", [])
        profile_name = getattr(module, "PROFILE_NAME", name)

        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower not in keyword_to_profiles:
                keyword_to_profiles[kw_lower] = []
            keyword_to_profiles[kw_lower].append(profile_name)

    conflicts = []
    for keyword, profile_names in keyword_to_profiles.items():
        if len(profile_names) > 1:
            conflicts.append(f"Keyword '{keyword}' used by: {', '.join(profile_names)}")

    return conflicts


def main() -> int:
    """Run validation."""
    parser = argparse.ArgumentParser(description="Validate profile modules")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-q", "--quiet", action="store_true", help="Only show errors")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    parser.add_argument("profile", nargs="?", help="Validate specific profile")
    args = parser.parse_args()

    profiles = discover_profiles()

    if args.profile:
        profiles = [(n, m, e) for n, m, e in profiles if n == args.profile]
        if not profiles:
            print(colorize(f"Profile not found: {args.profile}", Colors.RED))
            return 1

    total_errors = 0
    total_warnings = 0

    print(colorize(f"\n{'='*60}", Colors.BLUE))
    print(colorize("Profile Validation Report", Colors.BOLD))
    print(colorize(f"{'='*60}\n", Colors.BLUE))

    for name, module, load_error in profiles:
        if load_error:
            print(f"{colorize('SKIP', Colors.YELLOW)} {name}: {load_error}")
            continue

        is_valid, errors, warnings = validate_profile(module, args.verbose)
        profile_name = getattr(module, "PROFILE_NAME", name)

        if is_valid and not warnings:
            if not args.quiet:
                print(f"{colorize('PASS', Colors.GREEN)} {profile_name}")
        elif is_valid:
            status = colorize("WARN", Colors.YELLOW)
            print(f"{status} {profile_name}")
            for warning in warnings:
                print(f"  {colorize('⚠', Colors.YELLOW)} {warning}")
                total_warnings += 1
        else:
            status = colorize("FAIL", Colors.RED)
            print(f"{status} {profile_name}")
            for error in errors:
                print(f"  {colorize('✗', Colors.RED)} {error}")
                total_errors += 1
            for warning in warnings:
                print(f"  {colorize('⚠', Colors.YELLOW)} {warning}")
                total_warnings += 1

    # Check keyword conflicts
    if not args.quiet:
        print(colorize(f"\n{'-'*60}", Colors.BLUE))
        print(colorize("Cross-Profile Checks", Colors.BOLD))
        print(colorize(f"{'-'*60}\n", Colors.BLUE))

        conflicts = check_keyword_conflicts(profiles)
        if conflicts:
            print(colorize("Keyword conflicts:", Colors.YELLOW))
            for conflict in conflicts:
                print(f"  {colorize('⚠', Colors.YELLOW)} {conflict}")
                total_warnings += 1
        else:
            print(colorize("No keyword conflicts", Colors.GREEN))

    # Summary
    print(colorize(f"\n{'='*60}", Colors.BLUE))
    print(colorize("Summary", Colors.BOLD))
    print(colorize(f"{'='*60}\n", Colors.BLUE))

    valid_count = sum(1 for _, m, e in profiles if m and not e)
    print(f"Profiles validated: {valid_count}/{len(profiles)}")
    print(f"Errors: {total_errors}")
    print(f"Warnings: {total_warnings}")

    if args.strict and total_warnings > 0:
        print(colorize("\nFailed (strict mode, warnings present)", Colors.RED))
        return 1

    if total_errors > 0:
        print(colorize("\nFailed", Colors.RED))
        return 1

    print(colorize("\nPassed", Colors.GREEN))
    return 0


if __name__ == "__main__":
    sys.exit(main())
