from transformers import GPT2LMHeadModel, GPT2TokenizerFast
import re
from utils.text_utils import *


model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", add_prefix_space=True)


def sample_continuation(sentence_start: str, max_new_tokens: int = 20) -> str:
    inputs = tokenizer(sentence_start, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=1.0,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id,
    )

    new_tokens = output_ids[0, input_len:]
    continuation = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return continuation.strip()



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