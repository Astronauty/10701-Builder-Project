import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, tokenized_split_df_path):
        self.dataframe = pd.read_pickle(tokenized_split_df_path)

    def __getitem__(self, idx):
        return self.dataframe[idx]['en'], self.dataframe[idx]['fr']

    def __len__(self):
        return len(self.dataframe)
