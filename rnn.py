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
from data_loader import *
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, target_lang, dropout_prob=0.01):
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

    def train_epoch(self, data_pairs, encoder_optimizer, decoder_optimizer, criterion=bleu_score):
        """Runs an epoch of training on data data_pairs

                Args:
                    data_pairs (iterable): iterable of pairs of input and target tensors
                    encoder_optimizer (torch optimizer): optimizer of the encoder network
                    decoder_optimizer (torch optimizer): optimizer of the decoder network

                Returns:
                    float: average loss across the epoch
                """
        total_loss = 0
        for i in range(len(data_pairs[0])):
            input, target = data_pairs[0][i], data_pairs[1][i]
            input = torch.round(input).to(torch.int64)
            target = torch.round(target).to(torch.int64)
            input = input.unsqueeze(0)
            one_hot_target = F.one_hot(torch.round(target).to(torch.int64), num_classes=self.target_lang.n_words).float()
            target = target.unsqueeze(0)

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
                decoder_outputs[0], one_hot_target
            )
            print(loss.item())


            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_pairs)

    def train(self, data_points, n_epochs, learning_rate=0.001, print_every=100, plot_every=100, criterion=bleu_score):
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
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)

        for epoch in range(1, n_epochs + 1):
            print("epoch number:", epoch)
            loss = self.train_epoch(data_points, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('time: %s; epoch: (%d %d%%); average total loss per epoch: %.4f' % (timeSince(start, epoch / n_epochs),
                                             epoch, epoch / n_epochs * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)

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

    def test(self, data_pairs, criterion=bleu_score):
        """Evaluates the RNN on a test set

                        Args:
                            data_pairs (iterable): iterable of pairs of input and target tensors
                            criterion (function): the loss function

                        Returns:
                            float: average loss across the test data
                        """
        total_loss = 0
        for i in range(len(data_pairs[0])):
            input, target = data_pairs[0][i], data_pairs[1][i]
            input = torch.round(input).to(torch.int64)
            target = torch.round(target).to(torch.int64)
            input = input.unsqueeze(0)
            one_hot_target = F.one_hot(torch.round(target).to(torch.int64),
                                       num_classes=self.target_lang.n_words).float()
            target = target.unsqueeze(0)

            encoder_outputs, encoder_hidden = self.encoder(input)
            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target)

            loss = criterion(
                decoder_outputs[0], one_hot_target
            )
            print(loss)

            total_loss += loss

        return total_loss / len(data_pairs)

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

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)

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
    def __init__(self, hidden_dim, output_dim, max_output_length=100, SOS_token=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.LSTM = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.max_output_length = max_output_length
        self.SOS_token = SOS_token

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.SOS_token)
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
        output, hidden = self.LSTM(output, hidden)
        output = self.out(output)
        return output, hidden

def my_bleu_score(somewhat_hot_output, one_hot_target):
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

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('output_plot_300_epochs.png')

def create_rnn(used_abridged_data=True, max_seq_length=100, batch_size=32, shuffle=True, num_workers=0, hidden_dim=100,
               criterion=CrossEntropyLoss(), learning_rate=0.005, plot_every=1, print_every=1, n_epochs=50):
    abridge_tag = "_abridged"
    path = Path(f'data/EnFrDataset{abridge_tag}.pickle')

    if not path.exists():
        data = EnFrDataset(used_abridged_data=used_abridged_data, max_seq_length=max_seq_length)
        data.pickle_all_data()
    else:
        with open(path, 'rb') as handle:
            data = pickle.load(handle)

    # print(data.__getitem__(0))
    # print(data.__len__())

    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print(data)
    train_sequence_pair = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_sequence_pair[0].size()}")
    en_lang, fr_lang = data.en_lang, data.fr_lang

    input_dim, hidden_dim, output_dim = en_lang.n_words, hidden_dim, fr_lang.n_words

    rnn = RNN(input_dim, hidden_dim, output_dim, fr_lang)

    rnn.train(train_sequence_pair, n_epochs=n_epochs, criterion=criterion, learning_rate=learning_rate, plot_every=plot_every, print_every=print_every)

    return rnn

def save_rnn(rnn, file_name):
    rnn.save(file_name)

def load_rnn(file_name):
    with open(f"data/saved_rnns/{file_name}.pickle", 'rb') as handle:
        return pickle.load(handle)

def test_rnn(rnn, data_file_name, criterion=bleu_score):

    rnn.encoder.dropout = nn.Dropout(0)

    path = Path(f'data/{data_file_name}.pickle')

    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    print(len(data.data_pairs))

    train_dataloader = DataLoader(data, batch_size=1, shuffle=True, num_workers=0)
    print(train_dataloader)
    cumulative_error = 0
    for i in range(len(data.data_pairs)):
        print(i)
        train_sequence_pair = next(iter(train_dataloader))
        cumulative_error += rnn.test(train_sequence_pair, criterion=criterion)
    return cumulative_error/len(data.data_pairs)

"""rnn = create_rnn(n_epochs=10)
save_rnn(rnn, "10_epoch_on_abridged")"""


"""rnn = load_rnn("50_epoch_on_abridged")

print(test_rnn(rnn, "EnFrDataset_abridged", criterion=my_bleu_score))

print(torch.cuda.is_available())"""
"""candidate_corpus = [['My', 'full', 'pytorch', 'test']]
references_corpus = [[['My', 'full', 'pytorch', 'test', "sucks"]]]
print(bleu_score(candidate_corpus, references_corpus))
print(bleu_score([["1", "2", "4", "5"]], [[["1", "2", "4", "5", "6"]]]))"""







