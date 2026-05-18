#!/bin/bash

INPUT_FILE=$1
OUTPUT_DIR=$2
AFFIX=$3

echo "getting ngram surprisals"
time lm-zoo get-surprisals ngram $INPUT_FILE > $OUTPUT_DIR/ngram_$AFFIX.txt 2>/dev/null

echo "getting gpt2 surprisals"
time lm-zoo get-surprisals gpt2 $INPUT_FILE > $OUTPUT_DIR/gpt2_$AFFIX.txt 2>/dev/null

echo "getting grnn surprisals"
time lm-zoo get-surprisals GRNN $INPUT_FILE > $OUTPUT_DIR/grnn_$AFFIX.txt 2>/dev/null
