#!/bin/sh

CMD="./build/stream"
CSV_FILE="benchmark_results.csv"

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <tag> <arguments_for_command>"
    echo "Example: $0 v1 --mode=gpu samples/ACET.mp4"
    exit 1
fi

TAG="$1"
mode="$2"
video="$3"

if [ ! -f "$CSV_FILE" ]; then
    echo "Tag,Mode,Duration_ms" > "$CSV_FILE"
fi

START_TIME=$(date +%s%N)

echo "Running: $CMD $ARGS"
$CMD --mode=$mode $video

EXIT_CODE=$?

END_TIME=$(date +%s%N)

if [ $EXIT_CODE -ne 0 ]; then
    echo "Command failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

DURATION_NS=$((END_TIME - START_TIME))
DURATION_MS=$((DURATION_NS / 1000000))

echo "$TAG,$mode,$DURATION_MS" >> "$CSV_FILE"

echo "Benchmark saved: Tag='$TAG', Duration=${DURATION_MS}ms"
