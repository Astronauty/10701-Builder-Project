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
            Generate_train_test_split (bool): Check if generate train test split
            create_abridged: check if create abridged
            abridged_size: define abridge size
        Returns:
            
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
    """_summary_

    Args:
        ds: the pd dataset object
        en_tokenizer: English Tokenizer
        fr_tokenizer: French Tokenizer
    Returns:
        en_seq_len_list, en_seq_len_list: the list of sequences
    """
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
    """To organize the lang class objects.

    Args:
        lang: target lang object

    Returns:
        lang: organized target lang object
    """
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
    lang.index2word = index2word1
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
def add_list_of_tokens(batch_token_ids, reference: bool):
    """take in the batched tokens and turn it into list to feed into the BLEU function

    Args:
        batch_token_ids: the batched data
        reference: check if it is reference or not

    Returns:
        toks: the finished list of strings
    """
    toks = []

    custom_token_ids_to_remove = set([CustomTokens.SOS.value, CustomTokens.EOS.value, CustomTokens.PAD.value])
    for token_ids in batch_token_ids.tolist():
        temp = [str(token_id) for token_id in token_ids if token_id not in custom_token_ids_to_remove]

        if reference:
            toks.append([temp])
        else:
            toks.append(temp)

    return toks

def inference(model, src_data, tgt_data, out_seq_len):
    """Do inference to get the prediction BLEU score

    Args:
        model: the target model
        src_data: the source data
        tgt_data: the target data
        out_seq_len: the output sequence length

    Returns:
        output_pred:output of the prediction
        output_real:output of the real target data
    """
    model.eval()
    batch_size = src_data.shape[0]
    # initialize start of sentence
    y_init = torch.LongTensor([CustomTokens.SOS.value]).unsqueeze(0).cuda(1).view(1, 1)
    y_init = y_init.repeat(batch_size,1)

    # generate the encoder output from the encoder
    _, encoder_output = model(src_data, tgt_data)

    
    # inferencing
    for i in range(out_seq_len-1):
        # generate the mask for decoder
        _,tgt_mask = model.generate_mask(src_data, y_init)
        # get the embedding of the decoder input
        inf_emb = model.decoder_embedding(y_init)
        # added up with the positional encoding
        output_encoding_for_inference = inf_emb + model.positional_encoding.pe[:,:y_init.shape[1]]
        # get the decoder output and the probabilities of all the values
        decoder_output = model.pass_through_decoder(output_encoding_for_inference, encoder_output, tgt_mask)
        decoder_output = model.fc(decoder_output)
        # get the index of the final word with highest probabilities
        _, next_word = torch.max(
                decoder_output[:, y_init.shape[1] - 1 : y_init.shape[1],:], dim=2
            )
        # generate the final output
        y_init = torch.cat([y_init, next_word.view(batch_size,1)], dim=1)

    print(y_init.shape)
    # convert output from list to tokens
    output_pred = add_list_of_tokens((y_init), reference=False)
    # convert output ground truth from list to tokens
    output_real = add_list_of_tokens((tgt_data), reference=True)

        
    return output_pred, output_real

# from data_loader_full import Test_dataset
# from torchtext.data.metrics import bleu_score

# test_ds = Test_dataset("data/tokenized_test_en.csv","data/tokenized_test_fr.csv","data/en_lang_abridged_90.pickle", "data/fr_lang_abridged_90.pickle", sequence_length=60)

# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# src_vocab_size = test_ds.en_lang.n_words
# tgt_vocab_size = test_ds.fr_lang.n_words
# print("EN size:",src_vocab_size)
# print("FR size:",tgt_vocab_size)
# d_model = 512
# num_heads = 8
# d_ff = 512
# max_seq_length = 60
# encoder_blocks = 3
# assert((max_seq_length%(2**(encoder_blocks-1))==0)),"This shape is not compatible"
# assert((max_seq_length/(2**(encoder_blocks-1))!=0)),"This shape is not compatible"
# decoder_blocks = 8
# dropout = 0.1

# # initialize model
# f_transformer = Funnel_Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, decoder_blocks, encoder_blocks, d_ff, max_seq_length, dropout)
# f_transformer.load_state_dict(torch.load("model_ckpt/20231208-020407.pth")["model"])
# f_transformer = f_transformer.to(device)
# # initialize dataloader
# dataloader = DataLoader(test_ds, batch_size=256, shuffle=True, num_workers=6)

# all_pred = None
# all_real = None
# for i, it in enumerate(tqdm(dataloader)):
#     inputs, outputs = it
#     inputs = torch.squeeze(inputs)
#     outputs = torch.squeeze(outputs)
#     inputs = inputs.to(device)
#     outputs = outputs.to(device)
#     output_pred, output_real = inference(f_transformer,inputs, outputs,60)
#     if i == 0:
#         all_pred = output_pred
#         all_real = output_real
#     else:
#         all_pred.extend(output_pred)
#         all_real.extend(output_real)
#     print(bleu_score(all_pred,all_real))