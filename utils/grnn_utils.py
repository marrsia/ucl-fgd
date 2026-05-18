import os
import sys
import torch
import random
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, os.path.join(PROJECT_ROOT, "models/grnn"))
from model import RNNModel

VOCAB_PATH = os.path.join(PROJECT_ROOT, "models/grnn/vocab.txt")
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "models/grnn/grnn_state_dict.pt")

SPECIAL_TOKENS = {"</s>", "<s>", "<eos>", "<bos>"}

with open(VOCAB_PATH) as f:
    vocab = [line.strip() for line in f]

vocab_index = {word: i for i, word in enumerate(vocab)}

model = RNNModel('LSTM', ntoken=50001, ninp=650, nhid=650, nlayers=2, dropout=0.2)
state_dict = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True)
model.load_state_dict(state_dict)
model.eval()



def get_next_word_log_probs(context_ids, hidden):
    """
    Returns log probs over vocabulary given token id sequence and hidden state.
    """
    input_tensor = torch.tensor(context_ids).unsqueeze(1)  # (seq_len, 1)
    with torch.no_grad():
        output, hidden = model(input_tensor, hidden)
    log_probs = torch.log_softmax(output[-1, 0], dim=0).numpy()
    return log_probs, hidden


def sample_continuation(sentence_start, max_tokens=20):
    """
    Samples a continuation from the GRNN model until </s> or max_tokens.

    Args:
        sentence_start: string, the beginning of the sentence
        max_tokens: maximum number of new tokens to generate

    Returns:
        continuation string (not including the input)
    """
    words = sentence_start.lower().split()
    unk_id = vocab_index.get("<unk>", 0)
    token_ids = [vocab_index.get(w, unk_id) for w in words]

    hidden = model.init_hidden(1)

    # encode the sentence start
    input_tensor = torch.tensor(token_ids).unsqueeze(1)
    with torch.no_grad():
        _, hidden = model(input_tensor, hidden)

    continuation = []

    for _ in range(max_tokens):
        last_token_id = token_ids[-1]
        input_tensor = torch.tensor([[last_token_id]])
        with torch.no_grad():
            output, hidden = model(input_tensor, hidden)

        log_probs = torch.log_softmax(output[-1, 0], dim=0).numpy()
        probs = np.exp(log_probs)
        probs = probs / probs.sum()

        next_word = random.choices(vocab, weights=probs, k=1)[0]

        if next_word in SPECIAL_TOKENS:
            break

        continuation.append(next_word)
        token_ids.append(vocab_index.get(next_word, unk_id))
        
    return " ".join(continuation)