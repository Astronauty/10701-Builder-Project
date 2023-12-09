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
## to use this dataset use the following line
#1. the csv_file_x and csv_file_y should be the path to the data like the following
# ex. "data/tokenized_train_en.csv"
#2. the en_lang_path and fr_lang_path should be the path to the dictionaries like the following
# ex. "data/en_lang_80.pickle"
#3. the sequence_length can be what ever length you want
# ex.1000
#test_ds = Test_dataset(tokenized_x,tokenized_y,en_lang, fr_lang, sequence_length=100)


class Test_dataset(Dataset):
    def __init__(self, csv_file_x: str, csv_file_y: str, en_lang_path: str, fr_lang_path: str, sequence_length:int):
        self.en_lang_path = Path(en_lang_path)
        self.fr_lang_path = Path(fr_lang_path)
        self.csv_file_path_x = csv_file_x
        self.csv_file_path_y = csv_file_y
        self.tokenized_x = pd.read_csv(self.csv_file_path_x, sep='delimiter', header=None)
        self.tokenized_y = pd.read_csv(self.csv_file_path_y, sep='delimiter', header=None)
        self.sequence_length = sequence_length

        #check if dictionaries exists

        if self.en_lang_path.exists():
            with open(self.en_lang_path, 'rb') as handle:
                self.en_lang = pickle.load(handle)
        if self.fr_lang_path.exists():
            with open(self.fr_lang_path, 'rb') as handle:
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
    
    def list_of_tokens_to_list_of_words(self, list_of_tokens: torch.Tensor, test: bool,lang ):
        list_of_sentences = []
        for batch in list_of_tokens:
            list_of_words = []
            for token in batch:
                if token.item() == CustomTokens.EOS.value or token.item() == CustomTokens.PAD.value:
                    list_of_words.append(str(CustomTokens.EOS.value))
                    break
                # list_of_words.append(lang.index2word[token.item()])
                list_of_words.append(str(token.item()))
            if not test:
                list_of_sentences.append([list_of_words])
            else:
                list_of_sentences.append(list_of_words)
        return list_of_sentences
    
    # test_ds = Test_dataset("data/tokenized_train_en.csv","data/tokenized_train_fr.csv","data/en_lang_80.pickle", "data/fr_lang_80.pickle", sequence_length=100)

    # dataloader = DataLoader(test_ds, batch_size=32, shuffle=True, num_workers=6)

    # epoch_loss = []
    # total_loss = 0.0
    # for small_epoch in range(1000):
    #     f_transformer.train()
    #     small_epoch_count = 0
    #     np.random.seed()
    #     for it in dataloader:
    #         src_data, tgt_data = it
    #         src_data = src_data.to(device)
    #         src_data = torch.squeeze(src_data)
    #         tgt_data = tgt_data.to(device)
    #         tgt_data = torch.squeeze(tgt_data)
    #         optimizer.zero_grad()
    #         output= f_transformer(src_data, tgt_data[:, :-1])
    #         loss = loss_criterion(output[0].contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
    #         loss.backward()
    #         optimizer.step()
    #         total_loss += loss.item()
    #         del loss
    #         del output
    #     epoch_loss.append(total_loss)
    #     writer.add_scalar('training loss',
    #                     total_loss,
    #                         small_epoch)
    #     print(f"Small Epoch: {small_epoch+1}, Epoch Loss: {total_loss}")
    #     total_loss = 0.0