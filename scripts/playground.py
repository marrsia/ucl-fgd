import torch


import pandas as pd
from utils.text_utils import get_token_surprisal, get_region_token_ids, gpt2_tokenizer, get_gpt2_token_ids

surprisal_df = pd.read_csv("mock_data/grnn_output.csv")
stimuli_df = pd.read_csv("mock_data/table_data.csv")


# test get_token_surprisal
print(get_token_surprisal(surprisal_df, sentence_id=0, token_id=2))

# test get_region_token_ids

print(get_region_token_ids(stimuli_df, "comp", sentence_id=0, model_name="grnn"))
print(get_region_token_ids(stimuli_df, "ec1_object", sentence_id=1, model_name="grnn"))
print(get_region_token_ids(stimuli_df, "ec1_object", sentence_id=0, model_name="grnn"))  # should be None

print(get_region_token_ids(stimuli_df, "comp", sentence_id=0, model_name="gpt2"))
print(get_region_token_ids(stimuli_df, "ec1_object", sentence_id=1, model_name="gpt2"))
print(get_region_token_ids(stimuli_df, "ec1_object", sentence_id=0, model_name="gpt2"))  # should be None

for i in range(0, 7):
    print(get_gpt2_token_ids("I know what the lion devoured yesterday", i, gpt2_tokenizer))