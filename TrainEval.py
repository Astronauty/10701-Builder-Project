import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from data_loader import EnFrDataset
import matplotlib.pyplot as plt


class TrainEval:

    def __init__(self,
                 use_abridged_dataset,
                 num_epochs,
                 batch_size,
                 optimizer,
                 loss_function,
                 model: Module):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function

        # load dataset
        dataset = EnFrDataset(used_abridged_data=use_abridged_dataset,
                              max_seq_length=100)

        # init dataloaders
        self.train_dataloader = DataLoader(dataset=dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=1)

        # TODO: once dataloader has validation and test splits, load those here and update train loop accordingly.

        # move model to correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # init datastructures for storing intermediate train and validation losses
        self.epoch_train_losses = []
        self.epoch_val_losses = []

    def execute(self):
        self._run_all_epochs()

        self._display_loss_plot()

    def _run_all_epochs(self):
        # set model in training mode
        self.model.train()

        for e in range(1, self.num_epochs + 1):
            epoch_avg_train_loss = self._epoch_step()
            self.epoch_train_losses.append(epoch_avg_train_loss)

            print(f"e: {e}, train_loss: {round(epoch_avg_train_loss, 3)}")

        print(f"train_losses: {[round(epoch_avg_train_loss, 3) for epoch_avg_train_loss in self.epoch_train_losses]}")

    def _epoch_step(self):
        epoch_cumulative_train_loss = 0
        for en_token_ids, fr_token_ids in self.train_dataloader:
            # move input tensors to same device as model
            en_token_ids = en_token_ids.to(self.device)
            fr_token_ids = fr_token_ids.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(src=en_token_ids, tgt=fr_token_ids[:-1, :])

            loss = self.loss_function(output.reshape(-1, output.shape[-1]), fr_token_ids[1:, :].reshape(-1))

            loss.backward()
            self.optimizer.step()

            epoch_cumulative_train_loss += loss.item()

        epoch_avg_train_loss = epoch_cumulative_train_loss / len(self.train_dataloader)

        return epoch_avg_train_loss

    def _checkpoint(self):
        pass

    def _display_loss_plot(self):
        plt.plot(self.epoch_train_losses)
        plt.ylabel('train loss')
        plt.xlabel('epoch')
        plt.show()

    def _save_loss_plot(self):
        pass

    def _display_bleu_plot(self):
        pass

    def _save_bleu_plot(self):
        pass