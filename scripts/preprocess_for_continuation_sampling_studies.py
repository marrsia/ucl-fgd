import sys
from utils.text_utils import *

input_csv = sys.argv[1]
output_txt = sys.argv[2]
n_samples = int(sys.argv[3])

build_sentences_for_continuation_sampling(input_csv, output_txt, n_samples)