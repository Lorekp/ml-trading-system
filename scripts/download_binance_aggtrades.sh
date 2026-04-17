#!/bin/bash
# Download Binance USDT-M futures aggTrades for all bookTicker assets
# Monthly files, May 2023 - Mar 2024 (matching bookTicker range)
# Uses parallel curl (N_PARALLEL simultaneous downloads)

set -e

DATA_DIR="/c/Claude/ml-trading-system/data/binance-aggTrades"
ASSET_FILE="/tmp/assets.txt"
BASE_URL="https://data.binance.vision/data/futures/um/monthly/aggTrades"
N_PARALLEL=16
MONTHS="2023-05 2023-06 2023-07 2023-08 2023-09 2023-10 2023-11 2023-12 2024-01 2024-02 2024-03"

mkdir -p "$DATA_DIR"

# Generate all URLs
URL_FILE=$(mktemp)
TOTAL=0

while read -r SYMBOL; do
    SYMBOL_DIR="$DATA_DIR/$SYMBOL"
    mkdir -p "$SYMBOL_DIR"
    for MONTH in $MONTHS; do
        FILENAME="${SYMBOL}-aggTrades-${MONTH}.zip"
        OUTPATH="$SYMBOL_DIR/$FILENAME"
        if [ -f "$OUTPATH" ]; then
            continue  # skip already downloaded
        fi
        echo "$BASE_URL/$SYMBOL/$FILENAME|$OUTPATH" >> "$URL_FILE"
        TOTAL=$((TOTAL + 1))
    done
done < "$ASSET_FILE"

echo "=== Binance aggTrades Download ==="
echo "Assets: $(wc -l < "$ASSET_FILE")"
echo "Months: $(echo $MONTHS | wc -w)"
echo "Files to download: $TOTAL"
echo "Parallel: $N_PARALLEL"
echo ""

# Download function
download_one() {
    IFS='|' read -r URL OUTPATH <<< "$1"
    TMPPATH="${OUTPATH}.tmp"
    HTTP_CODE=$(curl -s -o "$TMPPATH" -w "%{http_code}" -L --connect-timeout 10 --max-time 300 "$URL" 2>/dev/null)
    if [ "$HTTP_CODE" = "200" ] && [ -s "$TMPPATH" ]; then
        mv "$TMPPATH" "$OUTPATH"
        BASENAME=$(basename "$OUTPATH")
        SIZE=$(stat -c%s "$OUTPATH" 2>/dev/null || stat -f%z "$OUTPATH" 2>/dev/null)
        SIZE_MB=$((SIZE / 1048576))
        echo "  OK ${BASENAME} (${SIZE_MB}MB)"
    else
        rm -f "$TMPPATH"
        # Don't print error for 404 (asset may not have futures for that month)
        if [ "$HTTP_CODE" != "404" ] && [ "$HTTP_CODE" != "000" ]; then
            echo "  FAIL $(basename "$OUTPATH") (HTTP $HTTP_CODE)"
        fi
    fi
}
export -f download_one

# Run parallel downloads
cat "$URL_FILE" | xargs -P "$N_PARALLEL" -I {} bash -c 'download_one "$@"' _ {}

rm -f "$URL_FILE"

# Count results
DOWNLOADED=$(find "$DATA_DIR" -name "*.zip" | wc -l)
TOTAL_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
echo ""
echo "=== Done ==="
echo "Downloaded: $DOWNLOADED files"
echo "Total size: $TOTAL_SIZE"
