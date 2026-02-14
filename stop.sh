#!/usr/bin/env bash
# Stop the SparkWorks Isaac Sim GUI
set -e

cd "$(dirname "$0")"

docker compose -f docker-compose.dev.yml --profile gui down

echo "SparkWorks GUI stopped."
