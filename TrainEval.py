
class TrainEval:

    def __init__(self,
                 use_abridged_dataset,
                 num_epochs,
                 batch_size,
                 optimizer,
                 loss_function):
        self.use_abridged_dataset = use_abridged_dataset
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss_function = loss_function

    def execute(self):
        pass

    def _run_all_epochs(self):
        pass

    def _epoch_step(self):
        pass

    def _epoch_sub_step(self):
        pass

    def _checkpoint(self):
        pass

    def _display_loss_plot(self):
        pass

    def _save_loss_plot(self):
        pass

    def _display_bleu_plot(self):
        pass

    def _save_bleu_plot(self):
        pass