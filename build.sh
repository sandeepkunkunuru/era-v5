#!/usr/bin/env bash
# Stage the publishable static site into dist/ — the course hub plus every
# session — keeping the dev-only verify/ harness and node_modules out.
set -euo pipefail
rm -rf dist
mkdir -p dist
cp index.html dist/
cp -R shared dist/
cp -R hub dist/
cp -R session-1 dist/
cp -R session-2 dist/
cp -R session-3 dist/
cp -R session-4 dist/
cp -R session-5 dist/
cp -R session-6 dist/
# drop any stray node_modules that shouldn't ship
find dist -name node_modules -type d -prune -exec rm -rf {} + 2>/dev/null || true
echo "Staged $(find dist -type f | wc -l | tr -d ' ') files into dist/"
