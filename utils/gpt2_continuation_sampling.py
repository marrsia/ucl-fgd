from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import re
from utils.text_utils import *


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)

def sample_continuations(sentence_start, n_samples=50, max_new_tokens=20):
    """
    Samples continuations from GPT-2 for a given sentence start.
    Runs until end-of-sentence punctuation, EOT token, or max_new_tokens.

    Args:
        sentence_start: string, the beginning of the sentence
        n_samples: number of continuations to sample
        max_new_tokens: maximum number of new tokens to generate

    Returns:
        list of continuation strings (not including the input)
    """
    input_ids = tokenizer(sentence_start, return_tensors="pt")["input_ids"]
    n_input_tokens = input_ids.shape[1]

    outputs = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_return_sequences=n_samples,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

    continuations = []
    for output in outputs:
        text = tokenizer.decode(output[n_input_tokens:], skip_special_tokens=True)
        # cut at first sentence-ending punctuation, keeping it
        match = re.search(r'[.!?]', text)
        if match:
            text = text[:match.end()]
        continuations.append(text)

    return continuations

def sample_all_continuations(input_csv, n_samples=50, max_new_tokens=20):
    """
    Samples continuations for all sentence starts and adds them to the dictionary.

    Args:
        sentence_starts: dict from build_sentence_starts_for_sampling
        n_samples: number of continuations per sentence
        max_new_tokens: max new tokens per continuation

    Returns:
        same dict with "continuations" added to each entry
    """
    sentence_starts = build_sentence_starts_for_sampling(input_csv)
    
    for idx, entry in sentence_starts.items():
        print(f"Sampling continuations for row {idx}: {entry['sentence_start']}")
        entry["continuations"] = sample_continuations(
            entry["sentence_start"],
            n_samples=n_samples,
            max_new_tokens=max_new_tokens
        )

    return sentence_starts