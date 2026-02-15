#!/usr/bin/env bash
# Run SparkWorks unit tests inside the Isaac Sim container.
#
# Usage:
#   ./run_tests.sh                                   # run ALL tests
#   ./run_tests.sh -v                                # verbose output
#   ./run_tests.sh TestExtrude                       # run one test class
#   ./run_tests.sh TestExtrude.test_basic_extrude    # run one test method
#
# Exit code: 0 = all tests passed, non-zero = failures.

set -e
cd "$(dirname "$0")"

TEST_SCRIPT="/isaac-sim/exts/sparkworks/sparkworks/tests/run_standalone.py"

echo "============================================"
echo "  SparkWorks Test Runner"
echo "============================================"
echo ""

# Check if container is already running
CONTAINER="isaacsim-cad"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER}$"; then
    echo "Using running container: ${CONTAINER}"
    docker exec "${CONTAINER}" /isaac-sim/python.sh "${TEST_SCRIPT}" "$@"
else
    echo "Starting container..."
    docker compose -f docker-compose.dev.yml run --rm --entrypoint /isaac-sim/python.sh sim "${TEST_SCRIPT}" "$@"
fi

EXIT=$?
echo ""
if [ $EXIT -eq 0 ]; then
    echo "All tests passed."
else
    echo "Some tests FAILED (exit code: $EXIT)"
fi
exit $EXIT
