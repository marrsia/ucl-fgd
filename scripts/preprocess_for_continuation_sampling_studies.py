import sys
from utils.text_utils import *

input_csv = sys.argv[1]
output_txt = sys.argv[2]

sentence_starts_dict = build_sentence_starts_for_sampling(input_csv)

with open(output_txt, "w") as f:
    for entry in sentence_starts_dict.values():
        f.write(entry["sentence_start"] + "\n")

print(f"Saved to {output_txt}")