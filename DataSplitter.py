import math

import pandas as pd


class DataSplitter:

    def __init__(self,
                 dataset_path,
                 abridge=False):
        self.abridge = abridge

        self.dataset = pd.read_csv(dataset_path,
                                   encoding="utf-8",
                                   keep_default_na=False)

        print('dataset length', len(self.dataset))

        self.split_proportion_dict = {
            'train': 0.8,
            'validation': 0.1,
            'test': 0.1
        }

    def split(self):
        prefix = "abridged_" if self.abridge else ""

        split_start_index = 0
        for split_name, proportion in self.split_proportion_dict.items():
            split_end_index = split_start_index + math.floor(len(self.dataset) * proportion)

            dataset_slice = self.dataset.iloc[split_start_index:split_end_index]

            print(len(dataset_slice))
            dataset_slice.to_pickle(f"data/splits/{prefix}{split_name}.pkl")

            split_start_index = split_end_index


if __name__ == '__main__':
    DataSplitter(dataset_path="data/en-fr-abridged.csv", abridge=True).split()
    # DataSplitter(dataset_path="data/en-fr.csv", abridge=False).split()
