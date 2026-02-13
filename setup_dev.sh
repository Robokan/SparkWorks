#!/usr/bin/env bash
# Setup script for the Parametric CAD Extension development environment.
#
# This script:
#   1. Creates the Isaac Sim cache directories on the host (needs sudo)
#   2. Builds the development Docker image
#   3. Prints instructions for running the container
#
# Usage: ./setup_dev.sh

set -euo pipefail

CACHE_ROOT="$HOME/docker/isaac-sim"

echo "=== SparkWorks — Dev Setup ==="
echo ""

# 1. Create host cache directories for Isaac Sim
# The Isaac Sim 5.1 container runs as UID 1234 — these dirs must be writable by that UID.
echo "Creating Isaac Sim cache directories (requires sudo)..."
sudo mkdir -p "$CACHE_ROOT/cache/main/ov"
sudo mkdir -p "$CACHE_ROOT/cache/main/warp"
sudo mkdir -p "$CACHE_ROOT/cache/computecache"
sudo mkdir -p "$CACHE_ROOT/config"
sudo mkdir -p "$CACHE_ROOT/data/documents"
sudo mkdir -p "$CACHE_ROOT/data/Kit"
sudo mkdir -p "$CACHE_ROOT/logs"
sudo mkdir -p "$CACHE_ROOT/pkg"
sudo chown -R 1234:1234 "$CACHE_ROOT"
echo "  Done."
echo ""

# 2. Build the dev Docker image
echo "Building development Docker image (this may take a while on first run)..."
docker compose -f docker-compose.dev.yml build
echo "  Done."
echo ""

# 3. Print usage instructions
echo "=== Setup Complete ==="
echo ""
echo "Run options:"
echo ""
echo "  Headless + Livestream (connect via WebRTC client):"
echo "    docker compose -f docker-compose.dev.yml up"
echo ""
echo "  Interactive bash inside the container:"
echo "    docker compose -f docker-compose.dev.yml run sim bash"
echo ""
echo "  GUI mode (requires X11, run 'xhost +local:' first):"
echo "    docker compose -f docker-compose.dev.yml --profile gui up"
echo ""
echo "Your extension source is live-mounted into the container at:"
echo "  /isaac-sim/exts/sparkworks"
echo ""
echo "Edit Python files locally — changes are picked up on extension reload."
echo "No rebuild needed for Python changes."
