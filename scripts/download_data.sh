#!/usr/bin/env bash
# Download the Lichess puzzle database (CC0) into data/raw/.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="$ROOT/data/raw/lichess_db_puzzle.csv.zst"
URL="https://database.lichess.org/lichess_db_puzzle.csv.zst"

mkdir -p "$ROOT/data/raw"
if [ -e "$TARGET" ]; then
    echo "Already present: $TARGET"
    exit 0
fi

echo "Downloading $URL"
curl -L --fail -o "$TARGET.part" "$URL"
mv "$TARGET.part" "$TARGET"
date -u +"Downloaded %Y-%m-%dT%H:%M:%SZ from $URL" > "$ROOT/data/raw/SOURCE.txt"
echo "Saved to $TARGET"
