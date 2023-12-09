from torch.utils.data import DataLoader

from DatasetSize import DatasetSize
from data_loader_full import Test_dataset


class DataLoaderProvider:

    def __init__(self,
                 dataset_size: DatasetSize,
                 batch_size,
                 max_sequence_length=40):
        self.dataloaders = {}

        for split in ['train', 'validation', 'test']:
            dataset = Test_dataset(csv_file_x=f"data/{dataset_size.value}/tokens/en_tokens_{split}.csv",
                                   csv_file_y=f"data/{dataset_size.value}/tokens/fr_tokens_{split}.csv",
                                   en_lang_path=f"data/{dataset_size.value}/vocabs/en_vocab.pkl",
                                   fr_lang_path=f"data/{dataset_size.value}/vocabs/fr_vocab.pkl",
                                   sequence_length=max_sequence_length)

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
