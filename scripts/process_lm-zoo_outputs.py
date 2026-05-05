import sys
from utils.text_utils import *

all_model_names = ["grnn", "ngram", "jrnn", "gpt2"]

stimuli_file = sys.argv[1]
files = sys.argv[2:]

for file in files:
    output = file.replace(".txt", ".csv")
    parse_lmzoo_output(file, output)

    model_name = next((m for m in all_model_names if m in file.lower()), None)
    if model_name is None:
        print(f"Warning: could not identify model from filename {file}, skipping.")
        continue

    results_file = file.replace(".txt", f"_stimuli_with_surprisals.csv")
    
    compute_region_surprisals(
        stimuli_csv=stimuli_file,
        surprisal_csv=output,
        output_csv=results_file,
        model_name=model_name,
        regions=list(RELEVANT_REGIONS)
    )