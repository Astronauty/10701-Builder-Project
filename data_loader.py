import torch
from torchtext.data.utils import get_tokenizer
from torch.utils.data.dataset import Dataset
import pandas as pd
import pickle
from tqdm import tqdm
import string
from pathlib import Path
from enum import Enum

from CustomToken import CustomToken
from Vocabulary import Vocabulary


class EnFrDataset(Dataset):

    def __init__(self,
                 max_seq_length,
                 used_abridged_data: bool):
        self.max_seq_length = max_seq_length

        self.dataset_path = Path("data/en-fr.csv")
        self.abridged_dataset_path = Path("data/en-fr-abridged.csv")

        self.dataset = self._load_dataset(used_abridged_data)

        self.en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        self.fr_tokenizer = get_tokenizer(tokenizer='spacy', language='fr_core_news_sm')

        self.process()
        pass

    def _load_dataset(self, use_abridged_data: bool):
        if use_abridged_data:
            self._create_abridged_dataset_if_necessary()

            dataset_path = pd.read_csv(self.abridged_dataset_path)
        else:
            dataset_path = pd.read_csv(self.dataset_path)

        dataset = pd.read_csv(dataset_path,
                              encoding="utf-8",
                              keep_default_na=False)

        return dataset

    def _create_abridged_dataset_if_necessary(self):
        if not self.abridged_dataset_path.exists():
            if not self.dataset_path.exists():
                raise FileNotFoundError("The full dataset does not exist. Please download it from "
                                        "https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset/data "
                                        "and place in the /data folder as 'en-fr.csv'.")

            abridged_dataset = pd.read_csv(self.dataset_path, nrows=5000)
            abridged_dataset.to_csv(self.abridged_dataset_path, index=False)


    def process(self):
        # Create abridged dataset if it doesnt exist and load either full or abridged data into self.ds
        # full_dataset_path = 'data/en-fr.csv'
        # abridged_dataset_path = 'data/en-fr-abridged.csv'
        self.full_dataset_path.parent.mkdir(parents=True, exist_ok=True)  # make datafolder if it doesn't exist

        # Check if the full dataset exists
        if not self.full_dataset_path.exists():
            raise FileNotFoundError(
                "The full dataset does not exist. Please download it from https://www.kaggle.com/datasets/dhruvildave/en-fr-translation-dataset/data and place in the /data folder.")

        # Create the abridged dataset if it does not exist
        if self.use_abridged_data and not self.abridged_dataset_path.exists():
            print("Creating abridged dataset...")
            full_dataset = pd.read_csv(self.full_dataset_path, nrows=5000)
            abridged_dataset = full_dataset.head(5000)
            abridged_dataset.to_csv(self.abridged_dataset_path, index=False)

        self.ds = pd.read_csv(self.abridged_dataset_path, il) if self.use_abridged_data else pd.read_csv(self.full_dataset_path,
                                                                                                encoding="utf-8",
                                                                                                keep_default_na=False)

        self._data_preprocessing()  # create data_pairs of lists of tokens
        pass

    def _data_preprocessing(self, eng_str="en", fr_str="fr"):
        """_summary_

        Args:
            en_tokenizer (_type_): _description_
            fr_tokenizer (_type_): _description_
            eng_str (str, optional): _description_. Defaults to "en".
            fr_str (str, optional): _description_. Defaults to "fr".
            data_pd (_type_, optional): _description_. Defaults to None.
            index_output (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        # initialize the language classes and get the data pairs (English, France)
        self.en_lang, self.fr_lang, self.data_pairs = self._read_lang(eng_str=eng_str, fr_str=fr_str,
                                                                      data_pd=self.ds)  # Initialize language objects
        print("Adding sentences to Vocabulary amd geting data pairs...")
        for i in tqdm(range(len(self.data_pairs))):  # create language dictionaries
            self.en_lang.addSentence(self.data_pairs[i][0].lower(), self.en_tokenizer, self.fr_tokenizer)
            self.fr_lang.addSentence(self.data_pairs[i][1].lower(), self.en_tokenizer, self.fr_tokenizer)

        self.data_pairs = self._string_data_to_tokens(self.data_pairs)  # converts sequence to tokens
        #  return en_lang, fr_lang, data_pairs
        pass

    def _read_lang(self, eng_str="en", fr_str="fr", data_pd=None):
        # return if the dataformat is wrong
        if type(self.ds) != pd.core.frame.DataFrame:
            raise TypeError("Wrong dataframe format!")

        print("Reading the dataframe and storing untokenized pairs...")
        pairs = [(self.ds[eng_str][i], self.ds[fr_str][i]) for i in tqdm(range(len(data_pd)))]

        # create the class objects of Vocabulary for English and French to count the 
        eng_lang = Vocabulary("en")
        fr_lang = Vocabulary("fr")
        return eng_lang, fr_lang, pairs

    def get_src_lang_size(self):
        return self.en_lang.n_words

    def get_tgt_lang_size(self):
        return self.fr_lang.n_words

    def pickle_data(self, nrows=None):
        en_lang, fr_lang, data_pairs = self._data_preprocessing(self.en_tokenizer, self.fr_tokenizer)

        abridge_tag = "_abridged" if self.use_abridged_data else ""

        with open(f'data/en_lang{abridge_tag}.pickle', 'wb') as handle:
            pickle.dump(en_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'data/fr_lang{abridge_tag}.pickle', 'wb') as handle:
            pickle.dump(fr_lang, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'data/data_pairs{abridge_tag}.pickle', 'wb') as handle:
            pickle.dump(data_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def pickle_all_data(self):
        abridge_tag = "_abridged" if self.use_abridged_data else ""

        with open(f'data/EnFrDataset{abridge_tag}.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def list_of_tokens_to_list_of_words(self, list_of_tokens: torch.Tensor, lang):
        list_of_sentences = []
        for batch in list_of_tokens:
            list_of_words = []
            for token in batch:
                if token.item() == CustomToken.EOS.value or token.item() == CustomToken.PAD.value:
                    list_of_words.append("EOS")
                    break
                list_of_words.append(lang.index2word[token.item()])
            list_of_sentences.append(list_of_words)
        return list_of_sentences

    def _string_data_to_tokens(self, data):
        """Create tokenized pairs of english and french sentences

        Args:
            data (_type_): Dictionary of english and french sentences

        Returns:
            _type_: _description_
        """
        tokenized_data = []
        print("Creating tokenized pairs of english and french sentences...")
        for i in tqdm(range(len(data))):
            tokenized_pair = (
            self._string_to_token_list(data[i][0], self.en_lang), self._string_to_token_list(data[i][1], self.fr_lang))
            tokenized_data.append(tokenized_pair)
        return tokenized_data

    def _string_to_token_list(self, sequence, lang):
        """Tokenize a sequence string in the given english/french language and return the list of tokens.

        Args:
            sequence (string): _description_
            lang (_type_): _description_
            en_tokenizer (_type_): _description_
            fr_tokenizer (_type_): _description_

        Returns:
            _type_: _description_
        """

        token_list = []
        if lang.lang == "en":
            words = self.en_tokenizer(sequence.lower())
        else:
            words = self.fr_tokenizer(sequence.lower())

        # truncate the word list if it exceeds the max allowed sequence length
        words = words[:self.max_seq_length - 2]  # -2 is to account for the appended SOS and EOS token

        token_list.append(CustomToken.SOS.value)
        for word in words:
            token_list.append(lang.word2index[word])

        token_list.append(CustomToken.EOS.value)

        # pad the remainder of the token list 
        while len(token_list) < self.max_seq_length:
            token_list.append(CustomToken.PAD.value)

        return token_list

    def __getitem__(self, idx):
        return torch.tensor(self.data_pairs[idx][0]), torch.tensor(self.data_pairs[idx][1])

    def __len__(self):
        return len(self.data_pairs)

# class CustomToken(Enum):
#     SOS = 0
#     EOS = 1
#     PAD = 2
#     UNK = 3
#
#
# class Vocabulary:
#
#     def __init__(self, tokenizer):
#         self.tokenizer = tokenizer
#         self.token2index = {}
#         self.token2count = {}
#         self.index2token = {
#             CustomToken.SOS.value: "SOS",
#             CustomToken.EOS.value: "EOS",
#             CustomToken.PAD.value: "PAD",
#             CustomToken.UNK.value: "UNK"
#         }
#
#         self.vocabulary_size = len(self.index2token)
#
#         # # put the punctuations inside the index2word dict
#         # for i in string.punctuation:
#         #     self.index2word[len(self.index2word)] = i
#
#         # # the total number of special words
#         # self.n_words = len(self.index2word)
#
#     def add_sentence(self, sentence):
#         for token in self.tokenizer(sentence):
#             self.add_token(token)
#
#     def add_token(self, token):
#         if token not in self.token2index:
#             self.token2index[token] = self.vocabulary_size
#             self.token2count[token] = 1
#             self.index2token[self.vocabulary_size] = token
#             self.vocabulary_size += 1
#         else:
#             self.token2count[token] += 1
