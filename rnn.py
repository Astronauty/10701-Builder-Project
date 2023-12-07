import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchtext.data.metrics import bleu_score
from torch.nn import MSELoss, CrossEntropyLoss
import math
import time
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from data_loader_full import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, target_lang, dropout_prob=0):
        """Initialises the RNN

                Args:
                    input_dim (int): number of english words in the dictionary
                    hidden_dim (int): dimensions of the hidden layers of the LSTMs
                    output_dim (int): number of french words in the dictionary
                    dropout_prob (float): the probability of omitting a word input during training

                Returns:
                none
                """
        super().__init__()
        # initiate the encoder and decoder
        self.encoder = EncoderRNN(input_dim, hidden_dim, dropout_prob=dropout_prob)
        self.decoder = DecoderRNN(hidden_dim, output_dim)
        self.target_lang = target_lang


    def train_epoch(self, data_loader, encoder_optimizer, decoder_optimizer, criterion=bleu_score, max_seq_length=60):
        """Runs an epoch of training on data data_pairs

                Args:
                    data_loader (iterable): iterable of pairs of input and target tensors
                    encoder_optimizer (torch optimizer): optimizer of the encoder network
                    decoder_optimizer (torch optimizer): optimizer of the decoder network

                Returns:
                    float: average loss across the epoch
                """
        total_loss = 0
        for data in tqdm(data_loader):
            input, target = data
            input = torch.squeeze(input).to(device)
            target= torch.squeeze(target).to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            encoder_outputs, encoder_hidden = self.encoder(input)
            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target)

            #print(decoder_outputs[0])
            #print(one_hot_target)

            # This is where I'm struggling to get the BLEU Score as a tensor rather than a float
            # I can only get BLEU to work with string inputs, but this and taking argmax in the
            # next line aren't differentiable.
            """maximizing_outputs = torch.topk(decoder_outputs, 1)[1].squeeze()
            target_array = target.tolist()
            maximizing_outputs_array = maximizing_outputs.tolist()
            print(maximizing_outputs_array)
            maximizing_outputs_strings = data_loader.list_of_tokens_to_list_of_words(maximizing_outputs_array, self.target_lang)
            print(target_array)
            target_strings = data_loader.list_of_tokens_to_list_of_words(target_array[0], self.target_lang)
            print([maximizing_outputs_strings], [[target_strings]])
            loss = criterion(
                [maximizing_outputs_strings], [[target_strings]]
            )"""

            # If someone can come up with a nice BLEU function, hopefully this should work:


            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)), target.view(-1)
            )


            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def train(self, data_loader, n_epochs, plot_name, learning_rate=0.001, print_every=100, plot_every=100, criterion=bleu_score):
        """Runs multiple epochs to train the RNN

                        Args:
                            data_pairs (iterable): iterable of pairs of input and target tensors
                            n_epochs (int): number of epochs to train
                            learning_rate (float): learning rate while training
                            print_every (int): how often (in terms of epochs) while training to print the performance across the last set of epochs
                            plot_every (int): how often to plot performance while training for the exported graph
                            criterion (function): the loss function being trained on.

                        Returns:
                            none
                        """
        start = time.time()
        plot_losses = []
        plot_validation_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        for epoch in range(1, n_epochs + 1):
            print("epoch number:", epoch)
            loss = self.train_epoch(data_loader, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('time: %s; epoch: (%d %d%%); average total loss per epoch: %.4f' %
                      (timeSince(start, (epoch) / (n_epochs)), epoch, epoch / n_epochs * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_validation_losses.append(test_rnn(self, criterion=CrossEntropyLoss()))
                plot_loss_total = 0

        showPlot(plot_losses, plot_name, 1, validation_points=plot_validation_losses)
        with open(f'data/{plot_name}_data.pickle', 'wb') as handle:
            pickle.dump({"plot_losses": plot_losses, "plot_validation_losses": plot_validation_losses, "time": time.time()}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        plot_losses

    def save(self, file_name):
        """Saves the RNN as a pickle file

                        Args:
                            file_name (string): the name of the file the RNN will be saved as

                        Returns:
                            none
                        """
        Path("data/saved_rnns").mkdir(parents=True, exist_ok=True)
        with open(f'data/saved_rnns/{file_name}.pickle', 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pass

    def test(self, data_loader, criterion=CrossEntropyLoss):
        """Evaluates the RNN on a test set

                        Args:
                            data_loader (iterable): iterable of pairs of input and target tensors
                            criterion (function): the loss function

                        Returns:
                            float: average loss across the test data
                        """
        total_loss = 0
        for data in tqdm(data_loader):
            input, target = data
            input = torch.squeeze(input).to(device)
            target = torch.squeeze(target).to(device)

            encoder_outputs, encoder_hidden = self.encoder(input)
            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target)

            loss = criterion(
                decoder_outputs.view(-1, decoder_outputs.size(-1)), target.view(-1)
            )

            total_loss += loss.item()

        return total_loss/len(data_loader)

class EncoderRNN(nn.Module):
    # The encoder LSTM of the RNN
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.01):
        """Initialises the encoder LSTM

                        Args:
                            input_dim (int): dimension of the input data - i.e. size of the English dictionary
                            hidden_dim (int): the size of the hidden LSTM layer
                            dropout_prob (float): how often a word is left out

                        Returns:
                            none
                        """
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, hidden_dim).to(device)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True).to(device)
        self.dropout = nn.Dropout(dropout_prob).to(device)
        self.to(device)

    def forward(self, input):
        """Forward pass on an input

                        Args:
                            input (iterable): iterable of input tensors

                        Returns:
                            output (list): a list of somewhat hot vectors, that is vectors supposed to emulate one-hot vectors hopefully close to the desired translation list of one hot vector

                        """
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_output_length=60, SOS_token=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim).to(device)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True).to(device)
        self.out = nn.Linear(hidden_dim, output_dim).to(device)
        self.max_output_length = max_output_length
        self.SOS_token = SOS_token
        self.to(device)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.SOS_token).to(device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_output_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if (target_tensor is not None) and (i < target_tensor.size()[0]):
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing

            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden

def my_bleu_score(outputs, targets):
    all_output_strings_lists = []
    all_targets_strings_lists = []
    for i in range(len(outputs)):
        somewhat_hot_output = [outputs[i]]
        target = targets[i]

        output_tokens = torch.topk(somewhat_hot_output, 1)[1].squeeze()
        target_tokens = torch.topk(one_hot_target, 1)[1].squeeze()
        output_token_list = output_tokens.tolist()
        output_token_str_list = [str(x) for x in output_token_list]
        target_token_list = target_tokens.tolist()
        target_token_str_list = [str(x) for x in target_token_list]
    return bleu_score([output_token_str_list], [[target_token_str_list]])



def asMinutes(secs):
    mins = math.floor(secs / 60)
    secs -= mins * 60
    return '%dm %ds' % (mins, secs)
def timeSince(since, percent):
    now = time.time()
    secs = now - since
    expected_secs = secs / (percent)
    time_left = expected_secs - secs
    return '%s (- %s)' % (asMinutes(secs), asMinutes(time_left))

def showPlot(points, plot_name, batches_per_epoch, validation_points=None):
    plt.figure()
    fig, ax = plt.subplots()
    x1 = np.arange(len(points) + 1)[1:]
    x = []
    for i in range(len(x1)):
        x.append(x1[i]/batches_per_epoch)
    plt.plot(x, points, label="training_error")
    if not validation_points == None:
        plt.plot(x, validation_points, label="validation_error")

    plt.xlabel("Batch number")
    plt.ylabel("Average cross entropy loss across batch")
    plt.savefig(f'{plot_name}.png')

def create_rnn(max_seq_length=60, batch_size=5000, shuffle=True, plot_name="no_plot_name", num_workers=0, hidden_dim=100,
               criterion=CrossEntropyLoss(), learning_rate=0.005, plot_every=1, print_every=1, n_epochs=50):
    data = Test_dataset("data/tokenized_train_en_abrdge.csv", "data/tokenized_train_fr_abrdge.csv", "data/en_lang_abridged_90.pickle",
                           "data/fr_lang_abridged_90.pickle", sequence_length=max_seq_length)

    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    en_lang, fr_lang = data.en_lang, data.fr_lang

    input_dim, hidden_dim, output_dim = en_lang.n_words, hidden_dim, fr_lang.n_words

    rnn = RNN(input_dim, hidden_dim, output_dim, fr_lang)

    rnn.train(train_dataloader, n_epochs, plot_name, criterion=criterion, learning_rate=learning_rate, plot_every=plot_every, print_every=print_every)

    return rnn

def save_rnn(rnn, file_name):
    rnn.save(file_name)

def load_rnn(file_name):
    with open(f"data/saved_rnns/{file_name}.pickle", 'rb') as handle:
        return pickle.load(handle)

def test_rnn(rnn, en_data_file_name="tokenized_val_en_abridge", fr_data_file_name="tokenized_val_fr_abridge", criterion=bleu_score, batch_size=256, max_seq_length=60):
    rnn.encoder.embedding.to(device)
    rnn.encoder.lstm.to(device)
    rnn.decoder.embedding.to(device)
    rnn.decoder.lstm.to(device)
    rnn.encoder.dropout = nn.Dropout(0)

    dataset = Test_dataset(f"data/{en_data_file_name}.csv", f"data/{fr_data_file_name}.csv",
                        "data/en_lang_abridged_90.pickle",
                        "data/fr_lang_abridged_90.pickle", sequence_length=max_seq_length)

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    error = rnn.test(test_dataloader, criterion=criterion)
    return error

for learning_rate in [0.001, 0.05, 0.01]:
    for hidden_dim in [50, 100, 150]:
        rnn = create_rnn(n_epochs=30, batch_size = 256, hidden_dim=100, plot_name=f"hidden_dim_{hidden_dim}_lr_{learning_rate}_plot")
        save_rnn(rnn, f"hidden_dim_{hidden_dim}_lr_{learning_rate}")


"""rnn = load_rnn("1_epoch_on_abridged")

print(test_rnn(rnn, criterion=CrossEntropyLoss()))
print(809.9715385437012)"""



"""candidate_corpus = [['My', 'full', 'pytorch', 'test']]
references_corpus = [[['My', 'full', 'pytorch', 'test', "sucks"]]]
print(bleu_score(candidate_corpus, references_corpus))
print(bleu_score([["1", "2", "4", "5"]], [[["1", "2", "4", "5", "6"]]]))"""







