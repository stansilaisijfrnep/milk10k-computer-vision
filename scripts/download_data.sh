#!/usr/bin/env bash
# Download and unpack the real MILK10k dataset (~345 MB) into data/milk10k/.
set -euo pipefail

URL="https://isic-archive.s3.amazonaws.com/dois/10.34970-648456/milk10k.zip"
DEST="data"

mkdir -p "$DEST"
if [ -d "$DEST/milk10k/images" ]; then
  echo "Dataset already present at $DEST/milk10k — nothing to do."
  exit 0
fi

echo "Downloading MILK10k (~345 MB)..."
curl -L --progress-bar -o "$DEST/milk10k.zip" "$URL"

echo "Unzipping..."
unzip -q -o "$DEST/milk10k.zip" -d "$DEST/milk10k"
rm "$DEST/milk10k.zip"

echo "Done. $(ls "$DEST/milk10k/images" | wc -l | tr -d ' ') images in $DEST/milk10k/images"
