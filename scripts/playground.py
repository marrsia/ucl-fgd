import sys
sys.path.insert(0, "models/grnn")
import torch
from model import RNNModel

with open("models/grnn/vocab.txt") as f:
    vocab = [line.strip() for line in f]
print(len(vocab))


# these are the hyperparameters from the filename: hidden650
model = RNNModel('LSTM', ntoken=len(vocab), ninp=650, nhid=650, nlayers=2, dropout=0.2)
state_dict = torch.load("models/grnn/grnn_state_dict.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()