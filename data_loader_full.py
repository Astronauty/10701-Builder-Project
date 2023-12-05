import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import string
from pathlib import Path
from enum import Enum
from data_loader import CustomTokens
import numpy as np

## Usage
#test_ds = Test_dataset("data/tokenized2_en.csv","data/tokenized2_fr.csv",en_lang, fr_lang, sequence_length=20)



class Test_dataset(Dataset):
    def __init__(self, csv_file_x: str, csv_file_y: str, en_lang_path, fr_lang_path, sequence_length:int):
        self.en_lang_path = Path(en_lang_path)
        self.fr_lang_path = Path(fr_lang_path)
        self.csv_file_path_x = csv_file_x
        self.csv_file_path_y = csv_file_y
        self.tokenized_x = pd.read_csv(self.csv_file_path_x, sep='delimiter', header=None)
        self.tokenized_y = pd.read_csv(self.csv_file_path_y, sep='delimiter', header=None)
        self.sequence_length = sequence_length

        #check if dictionaries exists

        if not self.en_lang_path.exists():
            with open('data/en_lang_90.pickle', 'rb') as handle:
                self.en_lang = pickle.load(handle)
        if not self.fr_lang_path.exists():
            with open('data/fr_lang_90.pickle', 'rb') as handle:
                self.fr_lang = pickle.load(handle)

    def __len__(self):
        return len(self.tokenized_x)
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        input_token = self.tokenized_x[0][index]
        input_token = np.array(input_token.split(","))
        input_token = input_token.astype(int)
        
        padded_input = torch.full((1,self.sequence_length), CustomTokens.PAD.value)
        if input_token.shape[0] > self.sequence_length:
            padded_input[0, :] = torch.Tensor(input_token.astype(int))[:self.sequence_length]
        else:
            padded_input[0, :input_token.shape[0]] = torch.Tensor(input_token.astype(int))

        output_token = self.tokenized_y[0][index]
        output_token = np.array(output_token.split(","))
        output_token = output_token.astype(int)
        padded_output = torch.full((1,self.sequence_length), CustomTokens.PAD.value)
        if output_token.shape[0] > self.sequence_length:
            padded_output[0, :] = torch.Tensor(output_token.astype(int))[:self.sequence_length]
        else:
            padded_output[0, :output_token.shape[0]] = torch.Tensor(output_token.astype(int))

        return padded_input, padded_output

    def get_src_lang_size(self):
        return self.en_lang.n_words
    
    def get_tgt_lang_size(self):
        return self.fr_lang.n_words
    
    def list_of_tokens_to_list_of_words(self, list_of_tokens: torch.Tensor, lang):
        list_of_sentences = []
        for batch in list_of_tokens:
            list_of_words = []
            for token in batch:
                if token.item() == CustomTokens.EOS.value or token.item() == CustomTokens.PAD.value:
                    list_of_words.append("EOS")
                    break
                list_of_words.append(lang.index2word[token.item()])
            list_of_sentences.append(list_of_words)
        return list_of_sentences