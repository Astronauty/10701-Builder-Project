import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import string

# tokenizers
en_tokenizer = get_tokenizer(tokenizer='spacy',language='en_core_web_sm')
fr_tokenizer = get_tokenizer(tokenizer='spacy',language='fr_core_news_sm')

# get the whole data from csv
whole_data = pd.read_csv("archive/en-fr.csv",encoding="utf-8", keep_default_na=False)

# Langs class
class Langs:
    def __init__(self, lang):
        # language name
        self.lang = lang
        # the dictionary for to get the index of word
        self.word2index = {}
        # the dictionary for the appear counts of each word
        self.word2count = {}
        # the dictionary to get the word through index (SOS: start of sentence, EOS: end of sentence)
        self.index2word = {0:"SOS", 1:"EOS"}
        # put the punctuations inside the index2word dict
        for i in string.punctuation:
            self.index2word[len(self.index2word)] = i
        # the total number of special words
        self.n_words = len(self.index2word)
    def addSentence(self, sentence):
        if self.lang == "en":
            tokens = en_tokenizer(sentence)
        else:
            tokens = fr_tokenizer(sentence)
        for word in tokens:
            self.addWord(word)
            
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def read_lang(eng_str="en", fr_str="fr", data_pd=None):
    print("reading lines....")
    # return if the dataformat is wrong
    if type(data_pd) != pd.core.frame.DataFrame:
        print("wrong data format!!")
        return
    # get the pairs
    pairs = [(data_pd[eng_str][i], data_pd[fr_str][i]) for i in tqdm(range(len(whole_data)))]
    
    # create the class objects of Langs for English and French to count the 
    eng_lang = Langs("en")
    fr_lang = Langs("fr")
    return eng_lang, fr_lang, pairs

def data_preprocessing(eng_str="en", fr_str="fr", data_pd=None):
    # initialize the language classes and get the data pairs (English, France)
    en_lang, fr_lang, data_pairs = read_lang(eng_str=eng_str, fr_str=fr_str, data_pd=data_pd)
    print("Got %s pairs" %len(data_pairs))
    # count the word occurences
    print("Creating Dictionary...")
    # only save the lower case words
    for i in tqdm(range(len(data_pairs))):
        en_lang.addSentence(data_pairs[i][0].lower())
        fr_lang.addSentence(data_pairs[i][1].lower())
    print("Counted words")
    print("en lenght:",en_lang.n_words)
    print("fr lenght:",fr_lang.n_words)
    return en_lang, fr_lang, data_pairs


en_lang, fr_lang, data_pairs = data_preprocessing(eng_str="en", fr_str="fr", data_pd=whole_data)


with open('en_lang.pickle', 'wb') as handle:
    pickle.dump(en_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('fr_lang.pickle', 'wb') as handle:
    pickle.dump(fr_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('data_pairs.pickle', 'wb') as handle:
    pickle.dump(data_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
