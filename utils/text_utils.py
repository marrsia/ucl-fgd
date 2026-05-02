import pandas as pd

SENTENCE_COLUMNS = [
    "mc_subject", "mc_verb", "comp", 
    "ec1_subject", "ec1_verb", "ec1_object", "continuation"
]

# Regions relevant for computing FGD surprisals.
RELEVANT_REGIONS = {
    "comp":  "complementiser",
    "ec1_object":  "gap",  # this might be missing
    "continuation": "post_gap"
}


def build_sentences(input_csv, output_txt):
    """
    Reads the CSV and writes one reconstructed sentence per line to a text file.
    """
    df = pd.read_csv(input_csv)

    with open(output_txt, "w") as f:
        for _, row in df.iterrows():
            words = []
            for col in SENTENCE_COLUMNS:
                value = str(row[col]).strip() if pd.notna(row[col]) else ""
                if value:
                    words.append(value)
            sentence = " ".join(words)
            f.write(sentence + "\n")

    print(f"Saved {input_csv} to {output_txt}")
    
def compute_region_indices(csv_path, region_cols=RELEVANT_REGIONS):
    df = pd.read_csv(csv_path)

    for region_name in region_cols.values():
        df[f"{region_name}_indices"] = None

    for idx, row in df.iterrows():
        word_index = 0

        for col in SENTENCE_COLUMNS:
            value = str(row[col]).strip() if pd.notna(row[col]) else ""

            if not value:
                continue

            col_words = value.split()
            col_start = word_index
            col_end = word_index + len(col_words) - 1

            if col in region_cols:
                region_name = region_cols[col]
                df.at[idx, f"{region_name}_indices"] = (col_start, col_end)

            word_index += len(col_words)

    df.to_csv(csv_path, index=False)
    print(f"Updated word indices for regions {region_cols.keys()} in {csv_path}.")
    
