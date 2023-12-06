import math
import torch
from torch.nn import Module

from DataLoaderProvider import DataLoaderProvider
import matplotlib.pyplot as plt


def unstage_plot():
    plt.clf()
    plt.close()


class TrainEval:

    def __init__(self,
                 dataloader_provider: DataLoaderProvider,
                 num_epochs,
                 optimizer,
                 loss_function,
                 model: Module,
                 model_shortname):
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function

        self.train_dataloader = dataloader_provider.get_train_dataloader()
        self.validation_dataloader = dataloader_provider.get_validation_dataloader()
        self.test_dataloader = dataloader_provider.get_test_dataloader()

        # move model to correct device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.model_shortname = model_shortname

        # init datastructures for storing intermediate train and validation losses
        self.epoch_train_losses = []
        self.epoch_validation_losses = []

    def _checkpoint(self, epoch, loss):
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }

        checkpoint_path = f"checkpoint/m-{self.model_shortname}_e-{epoch}_l-{loss}.pt"

        torch.save(checkpoint_dict, checkpoint_path)

    def execute(self):
        self._run_all_epochs()

        self._save_loss_plot()
        self._display_loss_plot()

    def _run_all_epochs(self):
        min_validation_loss = math.inf
        for e in range(1, self.num_epochs + 1):
            epoch_train_loss = self._epoch_step(train_mode=True)
            self.epoch_train_losses.append(epoch_train_loss)

            epoch_validation_loss = self._epoch_step(train_mode=False)
            self.epoch_validation_losses.append(epoch_validation_loss)

            print(
                f"e: {e}, train loss: {round(epoch_train_loss, 3)}, validation loss: {round(epoch_validation_loss, 3)}")

            # save checkpoint if model improved on validation set
            if epoch_validation_loss < min_validation_loss:
                self._checkpoint(e, epoch_validation_loss)
                min_validation_loss = epoch_validation_loss

    def _epoch_step(self, train_mode):
        if train_mode:
            self.model.train()
            dataloader = self.train_dataloader
        else:
            self.model.eval()
            dataloader = self.validation_dataloader

        epoch_cumulative_loss = 0
        for en_token_ids, fr_token_ids in dataloader:
            # move input tensors to same device as model
            en_token_ids = en_token_ids.to(self.device)
            fr_token_ids = fr_token_ids.to(self.device)

            self.optimizer.zero_grad()

            output = self.model(src=en_token_ids, tgt=fr_token_ids[:-1, :])

            loss = self.loss_function(output.reshape(-1, output.shape[-1]), fr_token_ids[1:, :].reshape(-1))

            if train_mode:
                loss.backward()
                self.optimizer.step()

            epoch_cumulative_loss += loss.item()

        epoch_avg_loss = epoch_cumulative_loss / len(dataloader)

        return epoch_avg_loss

    def _stage_loss_plot(self, train_values, validation_values):
        plt.plot(train_values, label=f"train")
        plt.plot(validation_values, label=f"validation")
        plt.xlabel('loss')
        plt.ylabel('epoch')
        plt.legend(loc="upper right")

    def _display_loss_plot(self):
        self._stage_loss_plot(self.epoch_train_losses, self.epoch_validation_losses)
        plt.show()
        unstage_plot()

    def _save_loss_plot(self):
        self._stage_loss_plot(self.epoch_train_losses, self.epoch_validation_losses)
        plt.savefig('loss.png')
        unstage_plot()

    def _display_bleu_plot(self):
        pass

    def _save_bleu_plot(self):
        pass
