#!/bin/bash

DIR=$1
OUTPUT=${2:-out.mp4}

function usage() {
  echo "$0 <directory>"
  exit 1
}

if [[ -z $DIR ]]; then
  usage
fi

ffmpeg -i $DIR/step_%06d.png -c:v libx264 -vf fps=30 $OUTPUT
