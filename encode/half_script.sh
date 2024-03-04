#!/usr/bin/env bash

set -eo pipefail

# ARGS:
#   $1 - Source file path
#   $2 - First half destination file path
#   $3 - Second half destination file path

length=$(ffprobe \
    -v warning \
    -print_format json \
    -show_entries stream=duration,codec_type \
    $1 | \
    jq -r '
        .streams[] |
        select(.codec_type=="video") |
        .duration
    ')

half_length=$(echo $length | awk '{ print $1/2 }')

ffmpeg \
    -v warning \
    -ss 0 -to ${half_length} \
    -i $1 \
    -c:a copy \
    -c:v copy \
    $2

ffmpeg \
    -v warning \
    -ss ${half_length} -to ${length} \
    -i $1 \
    -c:a copy \
    -c:v copy \
    $3
