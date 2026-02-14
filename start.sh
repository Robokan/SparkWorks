#!/usr/bin/env bash
# Start the SparkWorks Isaac Sim GUI
set -e

cd "$(dirname "$0")"

# Allow Docker containers to access the X11 display
xhost +local: > /dev/null 2>&1 || true

docker compose -f docker-compose.dev.yml --profile gui up -d

echo ""
echo "SparkWorks GUI starting..."
echo "Run './stop.sh' to shut it down."
echo "Run 'docker logs -f isaacsim-cad-gui' to view logs."
