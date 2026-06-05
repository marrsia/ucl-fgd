import pandas as pd
import spacy

nlp = spacy.load("en_core_web_sm")


def classify_continuation(full_sentence, continuation):
    doc = nlp(full_sentence.strip())
    cont = continuation.strip()
    for token in doc:
        if doc[token.i:].text.startswith(cont):
            if token.pos_ == "ADP":
                return "PP"
            if token.pos_ == "ADV":
                return "ADV"
            return "other"
    raise ValueError(f"Could not locate continuation '{cont}' in sentence '{full_sentence}'")


def tag_continuation_type(input_csv):
    df = pd.read_csv(input_csv)
    sentence_cols = [c for c in df.columns if c not in ("item", "condition")]
    df["full_sentence"] = df[sentence_cols].fillna("").agg(" ".join, axis=1).str.strip()
    df["continuation_type"] = df.apply(
        lambda row: classify_continuation(row["full_sentence"], row["continuation"]), axis=1
    )
    df.drop(columns=["full_sentence"], inplace=True)
    df.to_csv(input_csv, index=False)


def rename_conditions(input_csv):
    df = pd.read_csv(input_csv)

    condition_rename = {
        "that_gap" : "-wh_gap",
        "that_no-gap": "-wh_no_gap",
        "what_gap": "+wh_gap",
        "what_no-gap": "+wh_no_gap"
    }

    df["condition"] = df["condition"].map(condition_rename)
    df.to_csv(input_csv, index=False)
    
    
classify_continuation("data/stimuli/wilcox2022_embed4_object_gap_updated.csv")
