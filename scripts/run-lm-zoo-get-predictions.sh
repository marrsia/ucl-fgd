#!/bin/bash

INPUT_FILE=$1
OUTPUT_DIR=$2
AFFIX=$3

#echo "getting ngram predictions"
#time python3 scripts/ngram_get_predictions.py $INPUT_FILE $OUTPUT_DIR/ngram_$AFFIX.hdf5

#echo "getting gpt2 predictions"
#time lm-zoo get-predictions gpt2 $INPUT_FILE $OUTPUT_DIR/gpt2_$AFFIX.hdf5 

echo "getting grnn predictions"
time lm-zoo get-predictions GRNN $INPUT_FILE $OUTPUT_DIR/grnn_$AFFIX.hdf5