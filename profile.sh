#!/bin/sh

tag=$2
if [ "$1" = "nsys" ]; then
    echo "nsys $tag"
else
    echo "ncu $tag"
fi
