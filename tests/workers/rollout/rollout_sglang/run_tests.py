#!/usr/bin/env python3
"""Test runner for SGLang rollout tests.

This script provides a convenient way to run all SGLang-specific rollout tests
with appropriate configuration and options.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(
    verbose: bool = False,
    coverage: bool = False,
    html_coverage: bool = False,
    parallel: bool = False,
    pattern: str = None,
    markers: str = None,
    fail_fast: bool = False,
) -> int:
    """Run SGLang rollout tests with specified options.

    Args:
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        html_coverage: Generate HTML coverage report
        parallel: Run tests in parallel
        pattern: Pattern to filter test names
        markers: Pytest markers to filter tests
        fail_fast: Stop on first failure

    Returns:
        Exit code from pytest
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Test directory (current directory for this script)
    test_dir = Path(__file__).parent
    cmd.append(str(test_dir))

    # Add options based on arguments
    if verbose:
        cmd.extend(["-v", "-s"])

    if fail_fast:
        cmd.append("-x")

    if pattern:
        cmd.extend(["-k", pattern])

    if markers:
        cmd.extend(["-m", markers])

    # Coverage options
    if coverage or html_coverage:
        cmd.extend(["--cov=verl.workers.rollout.sglang_rollout", "--cov-report=term-missing"])

        if html_coverage:
            cmd.extend(["--cov-report=html:htmlcov_sglang"])

    # Parallel execution
    if parallel:
        try:
            import pytest_xdist  # noqa: F401

            cmd.extend(["-n", "auto"])
        except ImportError:
            print("Warning: pytest-xdist not installed, running sequentially")

    # Async support
    cmd.append("--asyncio-mode=auto")

    # Run the tests
    print(f"Running command: {' '.join(cmd)}")
    print(f"Test directory: {test_dir}")
    print("-" * 50)

    try:
        result = subprocess.run(cmd, cwd=test_dir.parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 130
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Run SGLang rollout tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py -v                 # Run with verbose output
  python run_tests.py -c                 # Run with coverage
  python run_tests.py -c --html          # Run with HTML coverage report
  python run_tests.py -p                 # Run in parallel
  python run_tests.py -k "test_init"     # Run tests matching pattern
  python run_tests.py -m "not slow"      # Run tests with specific markers
  python run_tests.py -v -c --html -x    # Verbose, coverage, HTML, fail-fast
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")

    parser.add_argument("-c", "--coverage", action="store_true", help="Enable coverage reporting")

    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report (implies --coverage)")

    parser.add_argument("-p", "--parallel", action="store_true", help="Run tests in parallel (requires pytest-xdist)")

    parser.add_argument("-k", "--pattern", type=str, help="Pattern to filter test names")

    parser.add_argument("-m", "--markers", type=str, help="Pytest markers to filter tests")

    parser.add_argument("-x", "--fail-fast", action="store_true", help="Stop on first failure")

    args = parser.parse_args()

    # If --html is specified, enable coverage
    if args.html:
        args.coverage = True

    exit_code = run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        html_coverage=args.html,
        parallel=args.parallel,
        pattern=args.pattern,
        markers=args.markers,
        fail_fast=args.fail_fast,
    )

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
