import pickle
from pathlib import Path

import pandas as pd
from pandas import DataFrame
from torchtext.data import get_tokenizer
from tqdm import tqdm

from Vocabulary import Vocabulary


class VocabularyBuilder:

    def __init__(self, dataframe: DataFrame, abridge=False, minimum_count=190):
        prefix = "abridged_" if abridge else ""

        self.en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        self.fr_tokenizer = get_tokenizer(tokenizer='spacy', language='fr_core_news_sm')

        en_vocab_path = f"data/vocabularies/{prefix}en_vocab.pkl"
        fr_vocab_path = f"data/vocabularies/{prefix}fr_vocab.pkl"

        if Path(en_vocab_path).exists() and Path(fr_vocab_path).exists():
            print("loading cached vocabularies")
            with open(f"data/vocabularies/{prefix}en_vocab.pkl", 'rb') as file:
                self.en_vocab = pickle.load(file)
            with open(f"data/vocabularies/{prefix}fr_vocab.pkl", 'rb') as file:
                self.fr_vocab = pickle.load(file)
        else:
            print("building vocabularies")
            self.en_vocab = Vocabulary(self.en_tokenizer)
            self.fr_vocab = Vocabulary(self.fr_tokenizer)

            for i, row in tqdm(dataframe.iterrows()):
                en_sentence = row['en']
                fr_sentence = row['fr']

                self.en_vocab.add_sentence(en_sentence)
                self.fr_vocab.add_sentence(fr_sentence)

            threshold = 1 if abridge else 190
            self.en_vocab.filter_out_rare_keys(threshold=threshold)
            self.fr_vocab.filter_out_rare_keys(threshold=threshold)

            with open(f"data/vocabularies/{prefix}en_vocab.pkl", 'wb') as file:
                pickle.dump(self.en_vocab, file, protocol=pickle.HIGHEST_PROTOCOL)
            with open(f"data/vocabularies/{prefix}fr_vocab.pkl", 'wb') as file:
                pickle.dump(self.fr_vocab, file, protocol=pickle.HIGHEST_PROTOCOL)


    def get_en_vocab(self):
        return self.en_vocab

    def get_fr_vocab(self):
        return self.fr_vocab


if __name__ == '__main__':
    abridged_train_df = pd.read_pickle(f"data/splits/abridged_train.pkl")

    VocabularyBuilder(dataframe=abridged_train_df, abridge=True)