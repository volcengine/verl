#!/usr/bin/env python3
"""Simple test runner for HTTP Server Engine tests.

This script provides a convenient way to run the HTTP server engine tests
with various options like coverage reporting and verbose output.
"""

import os
import subprocess
import sys


def run_tests(
    verbose: bool = False,
    coverage: bool = False,
    parallel: bool = False,
    test_pattern: str = None,
    html_report: bool = False,
):
    """Run the HTTP server engine tests with specified options.

    Args:
        verbose: Enable verbose output
        coverage: Enable coverage reporting
        parallel: Run tests in parallel
        test_pattern: Pattern to filter specific tests
        html_report: Generate HTML coverage report
    """
    # Base pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test file
    test_file = "test_http_server_engine.py"
    cmd.append(test_file)

    # Add options
    if verbose:
        cmd.extend(["-v", "-s"])

    if coverage:
        cmd.extend(["--cov=verl.workers.rollout.sglang_rollout.http_server_engine", "--cov-report=term-missing"])

        if html_report:
            cmd.append("--cov-report=html")

    if parallel:
        cmd.extend(["-n", "auto"])

    if test_pattern:
        cmd.extend(["-k", test_pattern])

    # Add asyncio mode for async tests
    cmd.append("--asyncio-mode=auto")

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 50)

    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTests interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run HTTP Server Engine tests")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-c", "--coverage", action="store_true", help="Enable coverage reporting")
    parser.add_argument("-p", "--parallel", action="store_true", help="Run tests in parallel")
    parser.add_argument("-k", "--pattern", type=str, help="Pattern to filter specific tests")
    parser.add_argument("--html", action="store_true", help="Generate HTML coverage report")

    args = parser.parse_args()

    # Check if we're in the right directory
    if not os.path.exists("test_http_server_engine.py"):
        print("Error: test_http_server_engine.py not found in current directory")
        print("Please run this script from tests/workers/rollout/ directory")
        return 1

    # Run tests
    return run_tests(
        verbose=args.verbose,
        coverage=args.coverage,
        parallel=args.parallel,
        test_pattern=args.pattern,
        html_report=args.html,
    )


if __name__ == "__main__":
    sys.exit(main())
