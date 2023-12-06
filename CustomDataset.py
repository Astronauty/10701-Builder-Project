import pandas as pd
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, tokenized_split_df_path):
        self.dataframe = pd.read_pickle(tokenized_split_df_path)
        print(len(self.dataframe))
        print(len(self.dataframe['en']))
        print(len(self.dataframe['fr']))

    def __getitem__(self, idx):
        return torch.tensor(self.dataframe.iloc[idx, self.dataframe.columns.get_loc("en")]), torch.tensor(self.dataframe.iloc[idx, self.dataframe.columns.get_loc("fr")])

    def __len__(self):
        return len(self.dataframe)
