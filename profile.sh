#!/bin/sh

stream="./build/stream"
if [ "$#" -eq 3 ]; then
    stream="$3"
fi

tag=$2
if [ "$1" = "nsys" ]; then
    nsys profile \
        --trace=cuda,cublas,osrt,nvtx \
        --stats=true \
        --output="$tag" \
        $stream --mode=gpu samples/ACETx4.mp4 --output=test.mp4
else
    sudo ncu \
        --set full \
        -o "$tag" \
        $stream --mode=gpu samples/ACETx4.mp4 --output=test.mp4
fi
