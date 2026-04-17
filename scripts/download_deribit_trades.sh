#!/bin/bash
# Download Deribit historical options + futures trades via public API
# BTC and ETH options give us implied volatility directly
# Uses pagination with parallel currency downloads

set -e

DATA_DIR="/c/Claude/ml-trading-system/data/deribit-trades"
BASE_URL="https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
CURRENCIES="BTC ETH"
# Match bookTicker range: May 2023 - Mar 2024
START_MS=1682899200000  # 2023-05-01
END_MS=1711929600000    # 2024-04-01

mkdir -p "$DATA_DIR"

echo "=== Deribit Historical Trades Download ==="
echo "Currencies: $CURRENCIES"
echo "Period: 2023-05-01 to 2024-03-31"
echo "Includes: options (with IV) + futures"
echo ""

download_currency() {
    CURRENCY=$1
    KIND=$2  # "option" or "future"
    OUTDIR="$DATA_DIR/${CURRENCY}_${KIND}"
    mkdir -p "$OUTDIR"

    # Download in daily chunks to avoid hitting pagination limits
    # 86400000 ms = 1 day
    CHUNK_MS=86400000
    CURRENT=$START_MS
    DAY_COUNT=0
    TRADE_COUNT=0

    while [ "$CURRENT" -lt "$END_MS" ]; do
        CHUNK_END=$((CURRENT + CHUNK_MS))
        if [ "$CHUNK_END" -gt "$END_MS" ]; then
            CHUNK_END=$END_MS
        fi

        # Convert ms to date for filename
        DATE=$(python -c "from datetime import datetime; print(datetime.utcfromtimestamp($CURRENT/1000).strftime('%Y-%m-%d'))")
        OUTFILE="$OUTDIR/${DATE}.json"

        if [ -f "$OUTFILE" ] && [ -s "$OUTFILE" ]; then
            CURRENT=$CHUNK_END
            DAY_COUNT=$((DAY_COUNT + 1))
            continue
        fi

        # Paginate within the day
        HAS_MORE="true"
        START_SEQ=""
        TMPFILE=$(mktemp)
        echo "[" > "$TMPFILE"
        FIRST=true
        PAGE_TRADES=0

        while [ "$HAS_MORE" = "true" ]; do
            if [ -z "$START_SEQ" ]; then
                RESPONSE=$(curl -s --connect-timeout 10 --max-time 30 \
                    "${BASE_URL}?currency=${CURRENCY}&kind=${KIND}&start_timestamp=${CURRENT}&end_timestamp=${CHUNK_END}&count=1000&sorting=asc" 2>/dev/null)
            else
                RESPONSE=$(curl -s --connect-timeout 10 --max-time 30 \
                    "${BASE_URL}?currency=${CURRENCY}&kind=${KIND}&start_timestamp=${CURRENT}&end_timestamp=${CHUNK_END}&count=1000&sorting=asc&start_seq=${START_SEQ}" 2>/dev/null)
            fi

            # Check for error
            ERROR=$(echo "$RESPONSE" | python -c "import sys,json; r=json.load(sys.stdin); print(r.get('error',{}).get('message',''))" 2>/dev/null)
            if [ -n "$ERROR" ] && [ "$ERROR" != "" ]; then
                echo "  API error for $CURRENCY $KIND $DATE: $ERROR — sleeping 5s"
                sleep 5
                continue
            fi

            # Extract trades and has_more
            TRADES=$(echo "$RESPONSE" | python -c "
import sys, json
r = json.load(sys.stdin)
result = r.get('result', {})
trades = result.get('trades', [])
has_more = result.get('has_more', False)
print(json.dumps({'trades': trades, 'has_more': has_more, 'count': len(trades)}))
" 2>/dev/null)

            COUNT=$(echo "$TRADES" | python -c "import sys,json; print(json.load(sys.stdin)['count'])" 2>/dev/null)
            HAS_MORE=$(echo "$TRADES" | python -c "import sys,json; print(str(json.load(sys.stdin)['has_more']).lower())" 2>/dev/null)

            if [ "$COUNT" = "0" ] || [ -z "$COUNT" ]; then
                break
            fi

            # Append trades to file
            if [ "$FIRST" = "true" ]; then
                echo "$TRADES" | python -c "import sys,json; [print(json.dumps(t)) for t in json.load(sys.stdin)['trades']]" >> "$TMPFILE"
                FIRST=false
            else
                echo "$TRADES" | python -c "import sys,json; [print(',' + json.dumps(t)) for t in json.load(sys.stdin)['trades']]" >> "$TMPFILE"
            fi

            PAGE_TRADES=$((PAGE_TRADES + COUNT))

            # Get next start_seq
            if [ "$HAS_MORE" = "true" ]; then
                START_SEQ=$(echo "$RESPONSE" | python -c "
import sys, json
trades = json.load(sys.stdin).get('result',{}).get('trades',[])
if trades: print(trades[-1].get('trade_seq', trades[-1].get('trade_id','')))
" 2>/dev/null)
                # Small delay to respect rate limits
                sleep 0.1
            fi
        done

        echo "]" >> "$TMPFILE"

        if [ "$PAGE_TRADES" -gt 0 ]; then
            mv "$TMPFILE" "$OUTFILE"
            TRADE_COUNT=$((TRADE_COUNT + PAGE_TRADES))
        else
            rm -f "$TMPFILE"
        fi

        DAY_COUNT=$((DAY_COUNT + 1))
        CURRENT=$CHUNK_END

        if [ $((DAY_COUNT % 30)) -eq 0 ]; then
            echo "  $CURRENCY $KIND: $DAY_COUNT days, $TRADE_COUNT trades so far"
        fi
    done

    echo "  $CURRENCY $KIND: DONE — $DAY_COUNT days, $TRADE_COUNT trades"
}

# Download options and futures for each currency
# Run BTC and ETH in parallel (2 currencies × 2 kinds = 4 streams)
# But be gentle on Deribit API — run sequentially per currency, parallel across currencies
for CURRENCY in $CURRENCIES; do
    echo "Starting $CURRENCY..."
    download_currency "$CURRENCY" "option" &
    # Small stagger to avoid hitting rate limits simultaneously
    sleep 2
    download_currency "$CURRENCY" "future" &
    sleep 2
done

wait

# Summary
echo ""
echo "=== Done ==="
TOTAL_FILES=$(find "$DATA_DIR" -name "*.json" -size +2c | wc -l)
TOTAL_SIZE=$(du -sh "$DATA_DIR" | cut -f1)
echo "Files: $TOTAL_FILES"
echo "Total size: $TOTAL_SIZE"
