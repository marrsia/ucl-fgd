#!/bin/bash

echo "getting ngram surprisals"
lm-zoo get-surprisals ngram ../mock_data/sentences.txt > ../mock_data/ngram_output.txt 2>&1

echo "getting gpt2 surprisals"
lm-zoo get-surprisals gpt2 ../mock_data/sentences.txt > ../mock_data/gpt2_output.txt 2>&1

echo "getting grnn surprisals"
lm-zoo get-surprisals GRNN ../mock_data/sentences.txt > ../mock_data/grnn_output.txt 2>&1

echo "getting jrnn surprisals"
lm-zoo get-surprisals JRNN ../mock_data/sentences.txt > ../mock_data/jrnn_output.txt 2>&1