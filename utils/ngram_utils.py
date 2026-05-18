import sys
import kenlm
import h5py
import numpy as np
import math
import os
import random

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(PROJECT_ROOT, "models/ngram/model.arpa")
vocab_path = os.path.join(PROJECT_ROOT, "models/ngram/vocab")

model = kenlm.Model(model_path)

with open(vocab_path) as f:
    vocab = [line.strip() for line in f]

vocab_index = {word: i for i, word in enumerate(vocab)}

def get_next_word_log_probs(context):
    """
    Returns array of log probs (base 10) over full vocabulary given context.
    """
    log_probs = np.array([
        list(model.full_scores(context + " " + word))[-1][0]
        for word in vocab
    ])
    return log_probs
  
SPECIAL_TOKENS = {"</s>", "<s>", "<eos>", "<bos>", "<unk>"}

def sample_continuation(sentence_start, max_tokens=20):
    context = sentence_start.lower()
    continuation = []

    # precompute mask for special tokens
    mask = np.array([1.0 if w not in SPECIAL_TOKENS else 0.0 for w in vocab])

    for _ in range(max_tokens):
        log_probs = get_next_word_log_probs(context)
        probs = np.array([10 ** lp for lp in log_probs])
        probs = probs * mask  # zero out special tokens
        probs = probs / probs.sum()

        next_word = random.choices(vocab, weights=probs, k=1)[0]
        continuation.append(next_word)
        context = context + " " + next_word

    return " ".join(continuation)

def old_sample_continuation(sentence_start, max_tokens=20):
    """
    Samples a continuation from the ngram model until </s> or max_tokens.

    Args:
        sentence_start: string, the beginning of the sentence
        max_tokens: maximum number of new tokens to generate

    Returns:
        continuation string (not including the input)
    """
    context = sentence_start.lower()
    continuation = []

    for _ in range(max_tokens):
        log_probs = get_next_word_log_probs(context)
        probs = np.array([10 ** lp for lp in log_probs])
        print(probs[:100])
        probs = probs / probs.sum()  # normalise to sum to 1

        next_word = random.choices(vocab, weights=probs, k=1)[0]

        if next_word in ["</s>", "<s>", "<eos>"]:
            break

        continuation.append(next_word)
        context = context + " " + next_word

    return " ".join(continuation)


def build_predictions(input_txt, output_hdf5):
    with open(input_txt) as f:
        sentences = [line.strip() for line in f if line.strip()]

    with h5py.File(output_hdf5, "w") as hf:
        # store vocabulary
        encoded_vocab = np.array([v.encode("utf-8") for v in vocab])
        hf.create_dataset("vocabulary", data=encoded_vocab)

        for i, sentence in enumerate(sentences):
            words = sentence.split()
            
            # get token ids
            token_ids = np.array([vocab_index.get(w.lower(), vocab_index.get("<unk>", 0)) for w in words])
            
            # get predictions for each position
            predictions = []
            for j in range(len(words)):
                context = " ".join(words[:j])
                log_probs = get_next_word_log_probs(context)
                predictions.append(log_probs)
            
            predictions = np.array(predictions)

            hf.create_dataset(f"sentence/{i}/tokens", data=token_ids)
            hf.create_dataset(f"sentence/{i}/predictions", data=predictions)

    print(f"Saved to {output_hdf5}")