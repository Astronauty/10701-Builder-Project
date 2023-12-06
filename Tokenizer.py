from pathlib import Path

import pandas as pd
from pandas import DataFrame
from torchtext.data import get_tokenizer

from CustomToken import CustomToken
from VocabularyBuilder import VocabularyBuilder


class Tokenizer:

    def __init__(self, vocab_dataframe, abridge, max_sequence_length):
        self.max_sequence_length = max_sequence_length
        self.en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
        self.fr_tokenizer = get_tokenizer(tokenizer='spacy', language='fr_core_news_sm')

        vocabulary_builder = VocabularyBuilder(dataframe=vocab_dataframe, abridge=abridge)

        self.en_vocab = vocabulary_builder.get_en_vocab()
        self.fr_vocab = vocabulary_builder.get_fr_vocab()

    def _truncation_sos_eos_padding(self, token_ids):
        # truncate
        token_ids = token_ids[:self.max_sequence_length - 2]  # -2 is to account for the appended SOS and EOS token

        # sos, eos
        token_ids = [CustomToken.SOS.value, *token_ids, CustomToken.EOS.value]

        # padding
        if len(token_ids) < self.max_sequence_length:
            diff = self.max_sequence_length - len(token_ids)
            padding = [CustomToken.PAD.value] * diff
            token_ids.extend(padding)

        return token_ids


    def tokenize_and_pad_row(self, row):
        en_tokens = self.en_tokenizer(row['en'])
        en_token_ids = [self.en_vocab.token2index[token] if token in self.en_vocab.token2index else CustomToken.UNK.value for token in en_tokens]
        en_token_ids = self._truncation_sos_eos_padding(en_token_ids)
        row['en'] = en_token_ids

        fr_tokens = self.fr_tokenizer(row['fr'])
        fr_token_ids = [self.fr_vocab.token2index[token] if token in self.fr_vocab.token2index else CustomToken.UNK.value for token in fr_tokens]
        fr_token_ids = self._truncation_sos_eos_padding(fr_token_ids)
        row['fr'] = fr_token_ids

        if len(en_token_ids) != len(fr_token_ids):
            print("ALERT!!!")
        return row


if __name__ == '__main__':
    vocab_df = pd.read_pickle("data/splits/abridged_train.pkl")
    tokenizer = Tokenizer(vocab_dataframe=vocab_df, abridge=True, max_sequence_length=50)

    for prefix in ['abridged_']:
        for split in ['train', 'validation', 'test']:
            dataframe_path = Path(f"data/splits/{prefix}{split}.pkl")

            dataframe = pd.read_pickle(dataframe_path)

            tokenized_dataframe : DataFrame = dataframe.apply(lambda row: tokenizer.tokenize_and_pad_row(row), axis=1)

            tokenized_dataframe.to_pickle(f"data/tokenized_splits/{prefix}{split}.pkl")

            print(tokenized_dataframe.to_string())


