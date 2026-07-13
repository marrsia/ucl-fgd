import pandas as pd
from transformers import GPT2TokenizerFast
from wordfreq import word_frequency

SENTENCE_COLUMNS = [
    "main_clause", "complementiser", "embedding_1" , "embedding_2", 
    "embedding_3", "embedding_4", "subject", "verb", "object" , "continuation"
]

SENTENCE_COLUMNS_FOR_SAMPLING_CONTINUATIONS_OBJECT_GAPS = [
    "main_clause", "complementiser", "embedding_1" , "embedding_2",  
    "embedding_3", "embedding_4", "subject", "verb"
]

SENTENCE_COLUMNS_FOR_SAMPLING_CONTINUATIONS_SUBJECT_GAPS = [
    "main_clause", "complementiser", "embedding_1" , "embedding_2",  
    "embedding_3", "embedding_4",
]

# Regions relevant for computing FGD surprisals.
RELEVANT_REGIONS = [
    "complementiser", #complementiser
    "object",  #object_gap"
    "subject", #subject_gap",
    "continuation" #post gap
]

gpt2_tokenizer =  GPT2TokenizerFast.from_pretrained("openai-community/gpt2", add_prefix_space=True)

def verify_data(input_csv):
    """
    Verifies structural integrity of a stimuli CSV.
    gap_type ('subject' or 'object') is inferred from the filename.

    Checks:
    1. Sentences are well-formed (spell-checked; proper nouns skipped)
    2. Within each sentence group, each content column has at most one distinct non-empty value
    3. wh/gap columns match the condition column; complementiser matches wh
    4. The gap filler column (subject or object) is empty iff gap=yes
    5. Number of non-empty embedding columns matches levels_of_embedding

    Args:
        input_csv: path to stimuli CSV

    Returns:
        True if all checks pass, False otherwise.
    """
    import os
    name = os.path.basename(input_csv).lower()
    if "object" in name:
        gap_type = "object"
    elif "subject" in name:
        gap_type = "subject"
    else:
        raise ValueError(f"Cannot infer gap_type from filename {os.path.basename(input_csv)!r}: expected 'subject' or 'object' in the name")

    df = pd.read_csv(input_csv)
    errors = []

    EMBEDDING_COLS = ["embedding_1", "embedding_2", "embedding_3", "embedding_4"]
    ALL_SENTENCE_COLS = [
        "main_clause", "complementiser", "embedding_1", "embedding_2",
        "embedding_3", "embedding_4", "subject", "verb", "object", "continuation"
    ]
    # Columns that vary by design within a sentence group
    VARYING_COLS = {"sentence_group", "wh", "gap", "condition", "levels_of_embedding", "complementiser"}
    CONTENT_COLS = [c for c in df.columns if c not in VARYING_COLS]
    gap_col = "subject" if gap_type == "subject" else "object"

    def cell(row, col):
        return str(row[col]).strip() if pd.notna(row[col]) else ""

    # Check 1: spell-check each word in the reconstructed sentence.
    # Proper nouns (words starting with uppercase mid-sentence) are skipped.
    from spellchecker import SpellChecker
    spell = SpellChecker()
    for idx, row in df.iterrows():
        parts = [cell(row, col) for col in ALL_SENTENCE_COLS if cell(row, col)]
        sentence = " ".join(parts)
        words = sentence.split()
        # skip first word (always capitalised) and mid-sentence proper nouns
        to_check = [w.lower().strip(".,;:!?\"'") for w in words[1:] if not w[0].isupper()]
        unknown = spell.unknown(to_check)
        if unknown:
            errors.append(f"Row {idx}: unrecognised word(s) {sorted(unknown)} in sentence: {sentence!r}")

    # Check 2: content columns should have at most one distinct non-empty value per group
    for group, group_df in df.groupby("sentence_group"):
        for col in CONTENT_COLS:
            non_empty = group_df[col].dropna()
            non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
            distinct = non_empty.astype(str).str.strip().unique()
            if len(distinct) > 1:
                errors.append(
                    f"Group {group}, column '{col}': inconsistent values: {list(distinct)}"
                )

    # Check 3: wh/gap columns match condition; complementiser matches wh
    for idx, row in df.iterrows():
        cond = cell(row, "condition")
        expected_wh = "yes" if cond.startswith("+wh") else "no"
        expected_gap = "yes" if cond.endswith("_gap") and not cond.endswith("_no_gap") else "no"
        if cell(row, "wh") != expected_wh:
            errors.append(
                f"Row {idx}: wh={row['wh']!r} doesn't match condition={cond!r} (expected {expected_wh!r})"
            )
        if cell(row, "gap") != expected_gap:
            errors.append(
                f"Row {idx}: gap={row['gap']!r} doesn't match condition={cond!r} (expected {expected_gap!r})"
            )
        comp = cell(row, "complementiser")
        if cell(row, "wh") == "yes" and comp != "who" and comp != "what":
            errors.append(f"Row {idx}: wh=yes but complementiser={comp!r} (expected 'who' or 'what')")
        elif cell(row, "wh") == "no" and comp != "that":
            errors.append(f"Row {idx}: wh=no but complementiser={comp!r} (expected 'that')")

    # Check 4: gap filler column is empty iff gap=yes
    for idx, row in df.iterrows():
        gap = cell(row, "gap")
        val = cell(row, gap_col)
        if gap == "yes" and val:
            errors.append(f"Row {idx}: gap=yes but {gap_col}={val!r} is not empty")
        elif gap == "no" and not val:
            errors.append(f"Row {idx}: gap=no but {gap_col} is empty")

    # Check 5: number of non-empty embedding columns matches levels_of_embedding
    for idx, row in df.iterrows():
        levels = int(row["levels_of_embedding"])
        for i, emb_col in enumerate(EMBEDDING_COLS, start=1):
            val = cell(row, emb_col)
            if i <= levels and not val:
                errors.append(
                    f"Row {idx}: levels_of_embedding={levels} but {emb_col} is empty"
                )
            elif i > levels and val:
                errors.append(
                    f"Row {idx}: levels_of_embedding={levels} but {emb_col}={val!r} is not empty"
                )

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print(f"\n{len(errors)} error(s) found.")
        return False

    print("All checks passed.")
    return True


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


def build_sentence_starts_for_sampling(input_csv, gap_type):
    """
    Builds sentence starts for continuation sampling from +wh_gap and 
    -wh_no_gap conditions only.

    Returns:
        dict of {row_number: {sentence_start, condition, levels_of_embedding}}
    """
    df = pd.read_csv(input_csv)
    df = df[df["condition"].isin(["+wh_gap", "-wh_no_gap"])]

    result = {}
    columns = SENTENCE_COLUMNS_FOR_SAMPLING_CONTINUATIONS_OBJECT_GAPS if gap_type == 'object' else SENTENCE_COLUMNS_FOR_SAMPLING_CONTINUATIONS_SUBJECT_GAPS
    for idx, row in df.iterrows():
        words = []
        for col in columns:
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
PADDED_MODELS = ["gpt2"]

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

    return token_ids



def get_multi_region_token_ids(df, regions, sentence_id, model_name):
    """
    Returns token id range spanning all tokens across a group of adjacent regions.
    Fails if regions are not adjacent in SENTENCE_COLUMNS.

    Args:
        df: dataframe read from original stimuli CSV
        regions: list of region names e.g. ["embedding_1", "embedding_2", "embedding_3"]
        sentence_id: the sentence number (0-indexed)
        model_name: model name string e.g. "grnn"

    Returns:
        (first word from first region, (start_token_id, end_token_id))
        or None if any region is empty
    """
    # check regions are adjacent in SENTENCE_COLUMNS
    indices = [SENTENCE_COLUMNS.index(r) for r in regions]
    if indices != list(range(min(indices), max(indices) + 1)):
        raise ValueError(f"Regions {regions} are not adjacent in SENTENCE_COLUMNS")

    # get token range for each region, skipping empty ones
    token_ranges = []
    

    for region in regions:
        result = get_region_token_ids(df, region, sentence_id, model_name)
        if result is None:
            continue
        token_range = result
        token_ranges.append(token_range)

    if not token_ranges:
        return None
    # span from start of first to end of last
    start = min(r[0] for r in token_ranges)
    end   = max(r[1] for r in token_ranges)

    return (start, end)


def get_range_mean_surprisal(surprisal_df, sentence_id, token_range):
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

def get_local_col(gap_type, condition):
        if gap_type == 'object':
            gap_col    = 'continuation'
            no_gap_col = 'object'
        elif gap_type == 'subject':
            gap_col = 'verb'
            no_gap_col = 'subject'
        
        if condition.endswith("no_gap"):
            return no_gap_col
        else:
            return gap_col

def compute_region_surprisals(stimuli_csv, surprisal_csv, output_csv, model_name, regions, gap_type):
    """
    For each sentence in the stimuli, computes the surprisal of the first token
    of each specified region and saves results to a CSV.
    Additionally saves local, semi-local and global surprisal in columns.

    Args:
        stimuli_csv: path to original stimuli CSV
        surprisal_csv: path to parsed lm-zoo surprisal CSV
        output_csv: path to save results
        model_name: model name string e.g. "grnn"
        regions: list of region names e.g. ["complementiser", "gap", "post_gap"]
        gap_type: can be 'subject' or 'object'
    """
    stimuli_df = pd.read_csv(stimuli_csv)
    surprisal_df = pd.read_csv(surprisal_csv)

    results = stimuli_df.copy()

    for region in regions:
        results[f"{region}_surprisal"] = None
        results[f"{region}_surprisal_mean"] = None
    results["semi_local_surprisal_mean"] = None
    results["global_surprisal_mean"] = None
    results["local_surprisal"] = None
    results["local_surprisal_mean"] = None
        
    
    for sentence_id, row in stimuli_df.iterrows():
        condition = row['condition']

        for region in regions:
            token_range = get_region_token_ids(stimuli_df, region, sentence_id, model_name)

            if token_range is None:
                continue

            results.at[sentence_id, f"{region}_surprisal"] = get_token_surprisal(
                surprisal_df, sentence_id, token_range[0]
            )
            results.at[sentence_id, f"{region}_surprisal_mean"] = get_range_mean_surprisal(
                surprisal_df, sentence_id, token_range
            )

        semi_local_regions = ["object", "continuation"] if gap_type == 'object' else ["subject", "verb", "object", "continuation"]

        results.at[sentence_id, "semi_local_surprisal_mean"] = get_range_mean_surprisal(
            surprisal_df, sentence_id, get_multi_region_token_ids(stimuli_df, semi_local_regions, sentence_id, model_name)
        )
        results.at[sentence_id, "global_surprisal_mean"] = get_range_mean_surprisal(
            surprisal_df, sentence_id, get_multi_region_token_ids(stimuli_df, regions, sentence_id, model_name)
        )

        local_col = get_local_col(gap_type, condition) + "_surprisal"
        results.at[sentence_id, "local_surprisal"] = results.at[sentence_id, local_col]
        results.at[sentence_id, "local_surprisal_mean"] = results.at[sentence_id, f"{local_col}_mean"]


    results.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
    
def add_region_base_frequency(output_with_surprisals_csv, regions, gap_type):
    """
        Compute base probabilities of: the first word in regions and full regions themselves.
        These do not yet have log applied to them. Save results to the same file.
        
        word_freq documentation states that: "This method of combining word frequencies implicitly assumes that you're 
        asking about words that frequently appear together. It's not multiplying the frequencies, because that would 
        assume they are statistically unrelated. So if you give it an uncommon combination of tokens, 
        it will hugely over-estimate their frequency"
        For this reason I will not be computing these for sections larger than just one region (no global or semi-local)
        
    """
    
    results_df = pd.read_csv(output_with_surprisals_csv)

    results = results_df.copy()
    
    for region in regions:
        results[f"{region}_frequency"] = None
        results[f"{region}_frequency_mean"] = None
    results["local_frequency"] = None
    results["local_frequency_mean"] = None
    
    for sentence_id, row in results_df.iterrows():
        condition = row['condition']
        for region in regions:
            if not pd.isnull(results.at[sentence_id, region]):
                first_word_freq = word_frequency( (results.at[sentence_id, region]).split()[0], 'en')
                mean_freq = word_frequency(results.at[sentence_id, region], 'en')
            
                results.at[sentence_id, f"{region}_frequency"] = first_word_freq
                results.at[sentence_id, f"{region}_frequency_mean"] = mean_freq

        local_region = get_local_col(gap_type, condition)
        
        results.at[sentence_id, "local_frequency"] = results.at[sentence_id, f"{local_region}_frequency"]
        results.at[sentence_id, "local_frequency_mean"] = results.at[sentence_id, f"{local_region}_frequency_mean"]

    
    results.to_csv(output_with_surprisals_csv, index=False)
    print(f"Saved to {output_with_surprisals_csv}") 
    
    