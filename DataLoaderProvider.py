import math

from torch.utils.data import DataLoader

from CustomDataset import CustomDataset


class DataLoaderProvider:

    def __init__(self,
                 abridge,
                 batch_size):
        self.dataloaders = {}

        for split in ['train', 'validation', 'test']:
            if abridge:
                dataset = CustomDataset(tokenized_split_df_path=f"data/tokenized_splits/abridged_{split}.pkl")
            else:
                dataset = CustomDataset(tokenized_split_df_path=f"data/tokenized_splits/{split}.pkl")

            self.dataloaders[split] = DataLoader(dataset=dataset,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=1)

    def get_train_dataloader(self):
        return self.dataloaders['train']

    def get_validation_dataloader(self):
        return self.dataloaders['validation']

    def get_test_dataloader(self):
        return self.dataloaders['test']
