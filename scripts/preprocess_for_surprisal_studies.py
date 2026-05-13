import sys
from utils.text_utils import *

input_csv = sys.argv[1]
output_txt = sys.argv[2]

build_sentences_for_surprisals(input_csv, output_txt)