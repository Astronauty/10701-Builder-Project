import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchtext.data.metrics import bleu_score
import math
import time
import torch.optim as optim
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import data_loader

class RNN:
    def __init__(self, input_dim, hidden_dim, output_dim, target_lang, dropout_p=0.1):
        super().__init__()
        self.encoder = EncoderRNN(input_dim, hidden_dim, dropout_p=dropout_p)
        self.decoder = DecoderRNN(hidden_dim, output_dim)
        self.target_lang = target_lang

    def train_epoch(self, data_pairs, encoder_optimizer, decoder_optimizer, criterion=bleu_score):
        total_loss = 0
        for pair in data_pairs:
            input, target = pair
            input = torch.tensor(input).unsqueeze(0)
            target = torch.tensor(target).unsqueeze(0)

            print(input.size())
            print(target.size())
            print("herererer")

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()


            encoder_outputs, encoder_hidden = self.encoder(input)
            decoder_outputs, _, _ = self.decoder(encoder_outputs, encoder_hidden, target)

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
                decoder_outputs, target
            )


            loss.backward()

            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

        return total_loss / len(data_pairs)

    def train(self, data_points, n_epochs, learning_rate=0.001, print_every=100, plot_every=100):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate)
        criterion = bleu_score

        for epoch in range(1, n_epochs + 1):
            loss = self.train_epoch(data_points, encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                             epoch, epoch / n_epochs * 100, print_loss_avg))

            if epoch % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        showPlot(plot_losses)


class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_p=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, max_output_length=50, SOS_token=0):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
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
            print(decoder_output)

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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

#data_loader.pickle_data(nrows=10000)

print(bleu_score([["1", "2"]],[["1", "3"]]))


print("about to do en_lang")
with open('en_lang.pickle', 'rb') as handle:
    en_lang = pickle.load(handle)
print("done it")
with open('fr_lang.pickle', 'rb') as handle:
    fr_lang = pickle.load(handle)
with open('data_pairs.pickle', 'rb') as handle:
    data_pairs = pickle.load(handle)

print(data_pairs[0])

input_dim, hidden_dim, output_dim = en_lang.n_words, 100, fr_lang.n_words

rnn = RNN(input_dim, hidden_dim, output_dim, fr_lang)

rnn.train(data_pairs, 10)




