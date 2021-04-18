import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
import os
from torch.utils.data import IterableDataset
import json
from graph_sort import create_graph

DATA_TRAIN_PATH = './data/train.data.jsonl'
LABEL_TRAIN_PATH = './data/train.label.json'


class TweetDataset(IterableDataset):

    def __init__(self, fn_data, fn_label, maxlen):
        # Store the contents of the file in a pandas dataframe
        self.fn_data = fn_data
        self.fn_data = fn_label

        # Initialize the BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.maxlen = maxlen

    def __iter__(self):

        with open(self.fn_data, encoding='utf-8') as f:

            for line in f:
                line = line.strip()
                if line.endswith(']'):
                    items = json.loads(line)
                    id_str = items[0]["id_str"]
                    idx_list = [pair[0] for pair in create_graph(items).topological_sort()]
                    tokens_list = []
                    for idx in idx_list:
                        username = items[idx]["user"]["name"]
                        text = items[idx]["text"]
                        current_sentence = username + ":" + text
                        tokens = ['[CLS]'] + tokenizer.tokenize(current_sentence) + ['[SEP]']  # automatically lowercase
                        if len(tokens) < maxlen_per_tweet:
                            padded_tokens = tokens + ['[PAD]' for _ in range(maxlen_per_tweet - len(tokens))]
                        else:
                            padded_tokens = tokens[:maxlen_per_tweet - 1] + ['[SEP]']
                        tokens_list.append(padded_tokens)
                    tokens_concat = [token for item in tokens_list for token in item]
                    seg_ids = []
                    acc = -1
                    for token in tokens_concat:
                        if token == '[CLS]':
                            acc += 1
                        seg_ids.append(acc)
                    tokens_ids = tokenizer.convert_tokens_to_ids(tokens_concat)
                    attn_mask = [1 if token != '[PAD]' else 0 for token in tokens_concat]
                    tokens_ids_tensor = torch.tensor(tokens_ids)
                    attn_mask_tensor = torch.tensor(attn_mask)
                    seg_ids_tensor = torch.tensor(seg_ids)
                    yield tokens_ids_tensor, attn_mask_tensor, seg_ids_tensor
                else:
                    raise KeyError("Lines are not in format!")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
maxlen_per_tweet = 60  # the max length of the sentence in the corpus


def main():
    pass


if __name__ == '__main__':
    main()
