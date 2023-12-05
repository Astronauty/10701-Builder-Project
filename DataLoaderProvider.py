import math

from torch.utils.data import DataLoader

from data_loader import EnFrDataset


class DataLoaderProvider:

    def __init__(self,
                 use_abridged_dataset,
                 batch_size,
                 max_sequence_length=100):
        # load dataset
        dataset = EnFrDataset(used_abridged_data=use_abridged_dataset,
                              max_seq_length=max_sequence_length)

        split_proportion_dict = {
            'train': 0.8,
            'validation': 0.1,
            'test': 0.1
        }

        self.dataloaders = {}

        # cumulative_proportion = 0
        split_start_index = 0
        for split_name, proportion in split_proportion_dict.items():
            split_end_index = split_start_index + math.floor(len(dataset) * proportion)

            dataset_slice = dataset[split_start_index:split_end_index]

            self.dataloaders[split_name] = DataLoader(dataset=dataset_slice,
                                                      batch_size=batch_size,
                                                      shuffle=False if split_name == 'test' else True,
                                                      num_workers=1)
            # cumulative_proportion += proportion
            split_start_index = split_end_index

    def get_train_dataloader(self):
        return self.dataloaders['train']

    def get_validation_dataloader(self):
        return self.dataloaders['validation']

    def get_test_dataloader(self):
        return self.dataloaders['test']
