import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import math
import copy
import spacy
import pathlib
from pathlib import Path
import csv

from funnel_transformer_conan import *
from data_loader import *


## Train Test Splitter, or split abridged dataset


class split_csv():
    def __init__(self, Generate_train_test_split:bool, create_abridged: bool, abridged_size: int):
        """_summary_

        Args:
            abridged (bool): Use the generated abridged dataset 

        Returns:
            _type_: _description_
        """
        self.Generate_train_test_split = Generate_train_test_split
        self.create_abridged = create_abridged
        self.abridged_size = abridged_size
        self.full_dataset_path = Path("data/en-fr.csv")
        self.abridged_dataset_path = Path("data/en-fr-training.csv")
        
        self.en_tokenizer = get_tokenizer(tokenizer='spacy',language='en_core_web_sm')
        self.fr_tokenizer = get_tokenizer(tokenizer='spacy',language='fr_core_news_sm')
        
        self.process()
        pass
    
    def process(self):
         # Create abridged dataset if it doesnt exist and load either full or abridged data into self.ds 
        # full_dataset_path = 'data/en-fr.csv'
        # abridged_dataset_path = 'data/en-fr-abridged.csv'
        self.full_dataset_path.parent.mkdir(parents=True, exist_ok=True) # make datafolder if it doesn't exist
        
        # Check if the full dataset exists
        if not self.full_dataset_path.exists():
            raise FileNotFoundError("The full dataset does not exist. Please download it from https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset/data and place in the /data folder.")
        
        # Create the abridged dataset if it does not exist
        if  self.Generate_train_test_split:      
            print("Creating train test dataset csvs...")
            print("reading full dataset...")
            full_dataset = pd.read_csv(self.full_dataset_path,encoding="utf-8", keep_default_na=False)
            if self.create_abridged:
                print("splitting abridged")
                abridged_dataset = full_dataset.head(self.abridged_size)
                print("splitting abridged train-test")
                abridged_dataset_train = abridged_dataset.head(int(self.abridged_size*0.9))
                abridged_dataset_val = abridged_dataset.tail(int(self.abridged_size*0.1))
                abridged_dataset_train.to_csv("data/en-fr-abridged-train.csv", index=False)
                abridged_dataset_val.to_csv("data/en-fr-abridged-val.csv", index=False)
            else:
                print("splitting fulldata train-test")
                training_len = int(len(full_dataset)*0.8)
                testing_len = int(len(full_dataset)-len(full_dataset)*0.9)

                abridged_dataset_training = full_dataset[:training_len]
                abridged_dataset_validation = full_dataset[training_len:training_len+testing_len]
                abridged_dataset_testing = full_dataset[training_len+testing_len:]
                del full_dataset

                print("Creating full data csvs...")
                abridged_dataset_training.to_csv("data/en-fr-train.csv", index=False)
                abridged_dataset_validation.to_csv("data/en-fr-val.csv", index=False)
                abridged_dataset_testing.to_csv("data/en-fr-test.csv", index=False)
            
        pass

# _ = split_csv(Generate_train_test_split=True,create_abridged=True, abridged_size=1000000)


#----------------------------------------------------------------------------------------------------


## Get the sequence length distribution and plot it


def get_the_sequence_distribution(ds,en_tokenizer,fr_tokenizer):
    en_seq_len_list = []
    fr_seq_len_list = []
    for i in tqdm(range(len(ds))):
        en_seq_len_list.append(len(en_tokenizer(ds['en'][i].lower())))
        fr_seq_len_list.append(len(fr_tokenizer(ds['fr'][i].lower())))

    return en_seq_len_list, en_seq_len_list
# en_tokenizer = get_tokenizer(tokenizer='spacy',language='en_core_web_sm')
# fr_tokenizer = get_tokenizer(tokenizer='spacy',language='fr_core_news_sm')

# ds = pd.read_csv("data/en-fr.csv",encoding="utf-8", keep_default_na=False)
# en_seq_list, fr_seq_list = get_the_sequence_distribution(ds,en_tokenizer, fr_tokenizer)

# import matplotlib.pyplot as plt
# from matplotlib.ticker import PercentFormatter
# import numpy as np

# en_seq_list = np.array(en_seq_list)
# fr_seq_list = np.array(fr_seq_list)
# bins = np.arange(0,110,10)
# fig, ax = plt.subplots(1,2,figsize=(15, 5))
# ax[1].hist([np.clip(en_seq_list, 0,100), np.clip(fr_seq_list, 0, 100)],cumulative=True,
#                             bins=bins, color=['#3782CC', '#AFD5FA'], weights=[np.full(len(en_seq_list),100)/len(en_seq_list), np.full(len(en_seq_list),100) / len(en_seq_list)], label=['English', 'French'])
# xlabels = bins[1:].astype(int).astype(str)
# xlabels[-1] += '+'

# N_labels = len(xlabels)
# ax[1].set_xlim([0, 100])
# ax[1].set_xticks(10 * np.arange(N_labels))
# ax[1].set_xticklabels(xlabels)
# ax[1].set_ylabel("Percentage(%)")
# ax[1].set_xlabel("Sequence Length")
# ax[1].legend()
# ax[1].set_title("CDF")
# fig.tight_layout()

# ax[0].hist([np.clip(en_seq_list, 0,100), np.clip(fr_seq_list, 0, 100)],
#                             bins=bins, color=['#3782CC', '#AFD5FA'], weights=[np.full(len(en_seq_list),100)/len(en_seq_list), np.full(len(en_seq_list),100) / len(en_seq_list)], label=['English', 'French'])
# xlabels = bins[1:].astype(int).astype(str)
# xlabels[-1] += '+'

# N_labels = len(xlabels)
# ax[0].set_xlim([0, 100])
# ax[0].set_xticks(10 * np.arange(N_labels))
# ax[0].set_xticklabels(xlabels)
# ax[0].set_ylabel("Percentage(%)")
# ax[0].set_xlabel("Sequence Length")
# ax[0].legend()
# ax[0].set_title("PDF")
# fig.tight_layout()


#----------------------------------------------------------------------------------------------------


## Create dictionaries for the dataset


def lang_organizer(lang):
    vocabs = list(lang.word2index.keys())
    for keys in vocabs:
        # check if the number of occurence in that key is small
        if lang.word2count[keys]<10:
            # the vocab keys appeared less than 10 times
            index = lang.word2index[keys]
            if index > 35:
                # remove all the stuff in the dictionaries and change the n_words
                del lang.word2index[keys]
                del lang.index2word[index]
                del lang.word2count[keys]
                lang.n_words -= 1
                
    vocabs = list(lang.index2word.values())
    index = [*range(len(vocabs))]
    word2index1 = {}
    index2word1 = {}
    i = 0
    for words in vocabs:
        if words not in word2index1:
            word2index1[words] = i
            index2word1[i] = words
            i += 1
    lang.word2index = word2index1
    lang.index2word1 = index2word1
    return lang

def read_lang(ds, eng_str="en", fr_str="fr"):
    # return if the dataformat is wrong
    if type(ds) != pd.core.frame.DataFrame:
        raise TypeError("Wrong dataframe format!")
        
    print("Reading the dataframe and storing untokenized pairs...")
    pairs = [(ds[eng_str][i], ds[fr_str][i]) for i in tqdm(range(len(ds)))]
    
    # create the class objects of Langs for English and French to count the 
    eng_lang = Langs("en")
    fr_lang = Langs("fr")
    return eng_lang, fr_lang, pairs

def string_to_token_list(sequence, lang):
        """Tokenize a sequence string in the given english/french language and return the list of tokens.

        Args:
            sequence (string): _description_
            lang (_type_): _description_
            en_tokenizer (_type_): _description_
            fr_tokenizer (_type_): _description_

        Returns:
            _type_: _description_
        """
        max_seq_length = 1000
        token_list = []
        if lang.lang == "en":
            words = en_tokenizer(sequence.lower())
        else:
            words = fr_tokenizer(sequence.lower())
            
        # truncate the word list if it exceeds the max allowed sequence length
        words = words[:max_seq_length - 2] # -2 is to account for the appended SOS and EOS token
        
        token_list.append(CustomTokens.SOS.value)
        for word in words:
            if word in lang.word2index:
                token_list.append(lang.word2index[word])
            else:
                token_list.append(CustomTokens.UNK.value)
        
        token_list.append(CustomTokens.EOS.value)
        
        # # pad the remainder of the token list 
        # while len(token_list) < max_seq_length:
        #     token_list.append(CustomTokens.PAD.value)
        
        return token_list

def string_data_to_tokens(data, en_lang, fr_lang, filename):
    """Create tokenized pairs of english and french sentences

    Args:
        data (_type_): Dictionary of english and french sentences

    Returns:
        _type_: _description_
    """
    tokenized_data = []
    fr_string = "_fr.csv"
    en_string = "_en.csv"
    print("Creating tokenized pairs of english and french sentences...")
    
    with open(filename+en_string, 'w') as csvfile1, open(filename+fr_string, 'w') as csvfile2:  
        # creating a csv writer object  
        csvwriter1 = csv.writer(csvfile1)  
        csvwriter2 = csv.writer(csvfile2)
        for i in tqdm(range(len(data))):
        # writing the fields  
            csvwriter1.writerow(string_to_token_list(data[i][0], en_lang))  
            csvwriter2.writerow(string_to_token_list(data[i][1], fr_lang))  

    return 1

def data_preprocessing(ds, training_data: bool, eng_str="en", fr_str="fr", ):
    """_summary_

    Args:
        en_tokenizer (_type_): _description_
        fr_tokenizer (_type_): _description_
        eng_str (str, optional): _description_. Defaults to "en".
        fr_str (str, optional): _description_. Defaults to "fr".
        data_pd (_type_, optional): _description_. Defaults to None.
        index_output (bool, optional): _description_cuda. Defaults to True.

    Returns:
        _type_: _description_
    """
    # initialize the language classes and get the data pairs (English, France)
    en_lang, fr_lang, data_pairs = read_lang(eng_str=eng_str, fr_str=fr_str, ds=ds) # Initialize language objects
    if training_data:
        print("Adding training set sentences to Langs amd geting data pairs...")
        for i in tqdm(range(len(data_pairs))): # create language dictionaries
            en_lang.addSentence(data_pairs[i][0].lower(), en_tokenizer, fr_tokenizer)
            fr_lang.addSentence(data_pairs[i][1].lower(), en_tokenizer, fr_tokenizer)
        # organize the langs
        print("Organizing Langs...")
        en_lang = lang_organizer(en_lang)
        fr_lang = lang_organizer(fr_lang)
        # save the langs into pickle files
        with open(f'data/en_lang_abridged_90.pickle', 'wb') as handle:
            pickle.dump(en_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'data/fr_lang_abridged_90.pickle', 'wb') as handle:
            pickle.dump(fr_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        print("Adding validations set sentences, so importing training set dictionaries...")
        with open(f'data/en_lang_abridged_90.pickle', 'rb') as handle:
            en_lang = pickle.load(handle)
        with open(f'data/fr_lang_abridged_90.pickle', 'rb') as handle:
            fr_lang = pickle.load(handle)

    print("Converting strings to tokens...")
    data_pairs = string_data_to_tokens(data_pairs,en_lang, fr_lang,"data/tokenized_test") # converts sequence to tokens
    print("Done Converting")
    #  return en_lang, fr_lang, data_pairs
    pass

# ds = pd.read_csv("data/full_data_set_splits/en-fr-test.csv",encoding="utf-8", keep_default_na=False)
# data_preprocessing(ds,training_data= False, eng_str="en", fr_str="fr")


#----------------------------------------------------------------------------------------------------


## Model Inference to get BLEU

