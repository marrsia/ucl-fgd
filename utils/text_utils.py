import pandas as pd
from transformers import GPT2TokenizerFast

SENTENCE_COLUMNS = [
    "main_clause", "complementiser", "embedding_1" , "embedding_2", 
    "embedding_3", "subject", "verb", "object" , "continuation"
]

SENTENCE_COLUMNS_FOR_SAMPLING_CONTINUATIONS = [
    "main_clause", "complementiser", "embedding_1" , "embedding_2",  
    "embedding_3", "subject", "verb"
]

# Regions relevant for computing FGD surprisals.
RELEVANT_REGIONS = [
    "complementiser", #complementiser
    "object",  #object_gap"
    "subject", #subject_gap",
    "continuation" #post gap
]

gpt2_tokenizer =  GPT2TokenizerFast.from_pretrained("openai-community/gpt2", add_prefix_space=True)

# TODO: write a data verifier 
# 1. check that sentences are all correct (grammar spelling etc), 
# 2. check that in each sentence group each column either always has the same thing or has nothing in it.
# 3. check that gap and filler columns match the condition column 
# 3. check that object column in empty if gap=yes and gap_type = object, same for subject
# 4. check that number of embeddings matches the condition


def build_sentences_for_surprisals(input_csv, output_txt):
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


def build_sentence_starts_for_sampling(input_csv):
    """
    Builds sentence starts for continuation sampling from +wh_gap and 
    -wh_no_gap conditions only.

    Returns:
        dict of {row_number: {sentence_start, condition, levels_of_embedding}}
    """
    df = pd.read_csv(input_csv)
    df = df[df["condition"].isin(["+wh_gap", "-wh_no_gap"])]

    result = {}

    for idx, row in df.iterrows():
        words = []
        for col in SENTENCE_COLUMNS_FOR_SAMPLING_CONTINUATIONS:
            value = str(row[col]).strip() if pd.notna(row[col]) else ""
            if value:
                words.append(value)
        
        result[idx] = {
            "sentence_start": " ".join(words),
            "condition": row["condition"],
            "levels_of_embedding": row["levels_of_embedding"]
        }

    return result
    
def parse_lmzoo_output(input_txt, output_csv):
    """
    Parses raw lm-zoo output file, extracts the tabular surprisal data,
    and saves it as a CSV. Ignores warnings and other non-tabular output.

    Args:
        input_txt: path to raw lm-zoo output file
        output_csv: path to save parsed CSV
    """
    rows = []
    
    with open(input_txt, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("sentence_id") or (len(line) > 0 and line[0].isdigit()):
                rows.append(line.split("\t"))

    df = pd.DataFrame(rows[1:], columns=rows[0])
    # Lm-zoo output is 1-indexed, so subtract 1 to get 0-indexing.
    df["sentence_id"] = df["sentence_id"].astype(int) - 1
    df["token_id"] = df["token_id"].astype(int) - 1
    df["surprisal"] = df["surprisal"].astype(float)

    df.to_csv(output_csv, index=False)
    print(f"Extracted output from {input_txt} and saved to {output_csv}.")
    


def get_token_surprisal(surprisal_df, sentence_id, token_id):
    """
    Returns surprisal for a specific token in a specific sentence.

    Args:
        surprisal_df: dataframe read from lm-zoo surprisal CSV
        sentence_id: the sentence number (1-indexed)
        token_id: the token number (1-indexed)

    Returns:
        surprisal as float, or None if not found
    """
    row = surprisal_df[
        (surprisal_df["sentence_id"] == sentence_id) & 
        (surprisal_df["token_id"] == token_id)
    ]

    if row.empty:
        print(f"Warning: no token found for sentence {sentence_id}, token {token_id}")
        return None

    return row.iloc[0]["surprisal"]



def get_gpt2_token_ids(sentence, start_word_idx, end_word_idx, tokenizer):
    """
    Returns token ids for a word at a given index in a sentence using GPT2 tokenizer.
    A word may map to multiple tokens.

    Args:
        sentence: full sentence as a string
        word_idx: 0-indexed word position in the sentence

    Returns:
        list of token ids for that word
    """
    
    words = sentence.split()
    encoding = tokenizer(words, is_split_into_words=True)
    
    word_ids = encoding.word_ids()
    
    # adding 1 to both tokens to match grnn and ngram which idx starting at 1
    token_positions = [pos for pos, word_id in enumerate(word_ids) if word_id >= start_word_idx and word_id <= end_word_idx]
    
    
    return (token_positions[0] + 1, token_positions[-1] + 1)


ONE_TOKEN_PER_WORD_MODELS = ["ngram", "grnn", "jrnn"]
PADDED_MODELS = ["jrnn", "gpt2"]

def get_region_token_ids(df, region, sentence_id, model_name):
    """
    Returns token ids for words in a specified region of interest for a given sentence.
    Currently only supports one-token-per-word models (ngram, grnn, jrnn).

    Args:
        stimuli_df: dataframe read from original stimuli CSV
        region_name: name of region e.g. "gap", "complementiser", "post_gap"
        sentence_id: the sentence number (0-indexed)
        model: model name string e.g. "grnn"

    Returns:
        (first word from region, [token range] )
    """
    if region not in SENTENCE_COLUMNS:
        raise ValueError(f"Unknown region '{region}'. "
                         f"Must be one of {SENTENCE_COLUMNS}")

    row = df.iloc[sentence_id]

    value = str(row[region]).strip() if pd.notna(row[region]) else ""
    if not value:
        return None

    # compute word offset of this region in the full sentence
    word_offset = 0
    for col in SENTENCE_COLUMNS:
        if col == region:
            break
        col_value = str(row[col]).strip() if pd.notna(row[col]) else ""
        if col_value:
            word_offset += len(col_value.split())

    region_words = value.split()
    first_word = region_words[0]
    
    if model_name.lower() in ONE_TOKEN_PER_WORD_MODELS:
        token_ids = (word_offset, word_offset + len(region_words))
    elif model_name.lower() == "gpt2":
        sentence_string = " ".join(
            str(row[col]).strip()
            for col in SENTENCE_COLUMNS
            if pd.notna(row[col]) and str(row[col]).strip()
        )
        token_ids = get_gpt2_token_ids(sentence_string, word_offset, word_offset + len(region_words), tokenizer = gpt2_tokenizer)
    else:
        raise NotImplementedError(f"Unknown model {model_name}.")

    # lm-zoo adds EOS tokens as boundries at the start and end of each sentence for some models.
    if model_name.lower() in PADDED_MODELS:
        start_id, end_id = token_ids
        token_ids = (start_id + 1, end_id + 1)

    return first_word, token_ids

def get_region_mean_surprisal(surprisal_df, sentence_id, token_range):
    """
    Returns mean surprisal over all tokens in a region.

    Args:
        surprisal_df: dataframe read from lm-zoo surprisal CSV
        sentence_id: the sentence number (1-indexed)
        token_range: (start, end) tuple of token ids (end exclusive)

    Returns:
        mean surprisal as float, or None if no tokens found
    """
    start, end = token_range
    rows = surprisal_df[
        (surprisal_df["sentence_id"] == sentence_id) &
        (surprisal_df["token_id"] >= start) &
        (surprisal_df["token_id"] < end)
    ]

    if rows.empty:
        return None

    return rows["surprisal"].mean()

def compute_region_surprisals(stimuli_csv, surprisal_csv, output_csv, model_name, regions):
    """
    For each sentence in the stimuli, computes the surprisal of the first token
    of each specified region and saves results to a CSV.

    Args:
        stimuli_csv: path to original stimuli CSV
        surprisal_csv: path to parsed lm-zoo surprisal CSV
        output_csv: path to save results
        model_name: model name string e.g. "grnn"
        regions: list of region names e.g. ["complementiser", "gap", "post_gap"]
    """
    stimuli_df = pd.read_csv(stimuli_csv)
    surprisal_df = pd.read_csv(surprisal_csv)

    results = stimuli_df.copy()

    for region in regions:
        results[f"{region}_surprisal"] = None
        results[f"{region}_surprisal_mean"] = None

    for sentence_id, _ in stimuli_df.iterrows():
        for region in regions:
            region_info = get_region_token_ids(stimuli_df, region, sentence_id, model_name)

            if region_info is None:
                continue

            _, token_range = region_info
            results.at[sentence_id, f"{region}_surprisal"] = get_token_surprisal(
                surprisal_df, sentence_id, token_range[0]
            )
            results.at[sentence_id, f"{region}_surprisal_mean"] = get_region_mean_surprisal(
                surprisal_df, sentence_id, token_range
            )
           
    results.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    