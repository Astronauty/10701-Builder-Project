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
from data_loader_full import *
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class RNN:
    # The class for the full network: has encoder and decoder networks and describes how they interact.
    def __init__(self, input_dim, hidden_dim, output_dim, target_lang, dropout_prob=0):
        """Initialises the RNN

                Args:
                    input_dim (int): number of english words in the dictionary
                    hidden_dim (int): dimensions of the hidden layers of the LSTMs
                    output_dim (int): number of french words in the dictionary
                    dropout_prob (float): the probability of omitting a word input during training. Ended up not using this in our final implementations.

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
                    data_loader (DataLoader): Contains batches of pairs of input and target tensors
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
                            data_loader (DataLoader): Contains batches of pairs of input and target tensors
                            n_epochs (int): number of epochs to train
                            plot_name (str): the name the plot is saved to
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

        showPlot(plot_losses, plot_name, validation_points=plot_validation_losses)
        with open(f'data/{plot_name}_data.pickle', 'wb') as handle:
            pickle.dump({"plot_losses": plot_losses, "plot_validation_losses": plot_validation_losses, "time": time.time() - start}, handle, protocol=pickle.HIGHEST_PROTOCOL)


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

    def test(self, data_loader, criterion=CrossEntropyLoss()):
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
            print(loss.item())

        return total_loss/len(data_loader)

class EncoderRNN(nn.Module):
    # The encoder LSTM of the RNN
    def __init__(self, input_dim, hidden_dim, dropout_prob=0):
        """Initialises the encoder LSTM

                        Args:
                            input_dim (int): dimension of the input data - i.e. size of the English dictionary
                            hidden_dim (int): the size of the hidden LSTM layer
                            dropout_prob (float): how often a word is left out (not used)

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
                            input (iterable): batch of input tensors

                        Returns:
                            output (list): a list of somewhat hot vectors, that is vectors supposed to emulate one-hot vectors hopefully close to the desired translation list of one hot vector

                        """
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.lstm(embedded)
        return output, hidden

class DecoderRNN(nn.Module):
    # The decoder RNN of the RNN class
    def __init__(self, hidden_dim, output_dim, max_output_length=60, SOS_token=0):
        """Initialises the decoder rnn

                                Args:
                                    hidden_dim (int): describes the n x n shape of the decoder LSTM. Must be the same as the encoder hidden_dim.
                                    output_dim (int): the number of french words in the dictionary
                                    max_output_length (int): the maximum length an output sentence can be
                                    SOS_token: The token first put into the decoder to generate a sentence from the initial context LSTM vector

                                Returns:
                                    None

                                """
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hidden_dim).to(device)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True).to(device)
        self.out = nn.Linear(hidden_dim, output_dim).to(device)
        self.max_output_length = max_output_length
        self.SOS_token = SOS_token
        self.to(device)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        """Forward pass on an input

                                Args:
                                    encoder_outputs (iterable): Outputs from the encoder network. Gives the size of the batch.
                                    encoder_hidden (iterable): The batch of context vectors from the encoder LSTM at the end of translation.
                                    target_tensor (tensor): If the rest of the output is known, this can be used to guide generation of the rest of the sentence given a 'bad start'. We did not use this in the final implementation.

                                Returns:
                                    decoder_outputs (iterable): the batch of outputs after a forward pass
                                    decoder_hidden (iterable): the batch of context vectors to be used as the networks hidden inputs for the next pass
                                    None: so the shapes match up while training

                                """
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long).fill_(self.SOS_token).to(device)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(self.max_output_length):
            decoder_output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if (target_tensor is not None) and (i < target_tensor.size()[0]):
                decoder_input = target_tensor[:, i].unsqueeze(1)

            else:
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None

    def forward_step(self, input, hidden):
        """Forward step on an input

                                Args:
                                    input (iterable): batch of input tensors (normally from last step)
                                    hidden (iterable): batch of hidden LSTM input tensors (normally from last step)

                                Returns:
                                    output (iterable): batch of output 'tokens'
                                    hidden (iterable): the hidden LSTM outputs for the next forward step

                                """
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        return output, hidden

def my_bleu_score(outputs, targets):
    """Implements the bleu_score metric on translated outputs from the rnn.

                            Args:
                                outputs (iterable): batch of tensors output from the network - representing a tokenised translation
                                targets (iterable): the tokenised target vector

                            Returns:
                                score (float): the average BLEU score of the network output vs the targets.

                            """
    all_output_strings_lists = []
    all_targets_strings_lists = []
    targets_list = targets.tolist()
    for i in range(len(outputs)):
        somewhat_hot_output = outputs[i].unsqueeze(0)
        target_token = str(targets_list[i])
        output_token = str(torch.topk(somewhat_hot_output, 1)[1].squeeze().item())
        if not output_token in ["0", "1", "2"]:
            all_output_strings_lists.append(output_token)
        if not target_token in ["0", "1", "2"]:
            all_targets_strings_lists.append(target_token)
    return torch.tensor(bleu_score([all_output_strings_lists], [[all_targets_strings_lists]]))



def asMinutes(secs):
    """Used for returning time used in computations

                            Args:
                                secs (float):

                            Returns:
                                time (list):

                            """
    mins = math.floor(secs / 60)
    secs -= mins * 60
    return '%dm %ds' % (mins, secs)
def timeSince(since, percent):
    """Returns the time since the 'since' time, and the expected time left.

                            Args:
                                since (float):
                                percent (float):

                            Returns:
                                time taken, time left (float):

                            """
    now = time.time()
    secs = now - since
    expected_secs = secs / (percent)
    time_left = expected_secs - secs
    return '%s (- %s)' % (asMinutes(secs), asMinutes(time_left))

def showPlot(points, plot_name, validation_points=None):
    """Plots and saves a graph of training error vs epoch.

                            Args:
                                points (list): list of training error values at the end of each epoch
                                plot_name (str): determines the name of the plot when saved
                                validation_points (list): if validation error is plotted also, this is a list of validation error values at the end of each epoch

                            Returns:
                                None

                            """
    plt.figure()
    fig, ax = plt.subplots()
    x1 = np.arange(len(points) + 1)[1:]
    x = []
    for i in range(len(x1)):
        x.append(x1[i])
    plt.plot(x, points, label="Training error")
    if not validation_points == None:
        plt.plot(x, validation_points, label="Validation error")

    plt.xlabel("Epoch number")
    plt.ylabel("Average cross entropy loss across epoch")
    plt.savefig(f'{plot_name}.png')

def create_rnn(max_seq_length=60, batch_size=256, shuffle=True, plot_name="no_plot_name", num_workers=0, hidden_dim=100,
               criterion=CrossEntropyLoss(), learning_rate=0.005, plot_every=1, print_every=1, n_epochs=50):
    """Creates an rnn with the parameters described, as well as a plot of the training/validation error

                            Args:
                                max_seq_length (float): The maximum length of a sentence input
                                batch_size (int): size of DataLoader batches
                                shuffle (Boolean): determines whether the DataLoader shuffles the data
                                plot_name (str): specifies the filename of the plot of training errors
                                hidden_dim (int): the n x n dimensions of the hidden layers of the encoder and decoder
                                criterion (function): the loss function while training
                                learning_rate (float): the learning rate while training
                                plot_every (int): determines how often to plot errors
                                print_every (int): determines when to print progress reports
                                n_epochs (int): the number of training epochs

                            Returns:
                                rnn (RNN): the trained rnn

                            """
    data = Test_dataset("data/tokenized_train_en_abrdge.csv", "data/tokenized_train_fr_abrdge.csv", "data/en_lang_abridged_90.pickle",
                           "data/fr_lang_abridged_90.pickle", sequence_length=max_seq_length)

    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    en_lang, fr_lang = data.en_lang, data.fr_lang

    input_dim, hidden_dim, output_dim = en_lang.n_words, hidden_dim, fr_lang.n_words

    rnn = RNN(input_dim, hidden_dim, output_dim, fr_lang)

    rnn.train(train_dataloader, n_epochs, plot_name, criterion=criterion, learning_rate=learning_rate, plot_every=plot_every, print_every=print_every)

    return rnn

def save_rnn(rnn, file_name):
    """saves the rnn to the file name
                            Args:
                                rnn (RNN): the rnn
                                file_name (str): the file_name

                            Returns:
                                None
                            """
    rnn.save(file_name)

def load_rnn(file_name):
    """Returns the time since the 'since' time, and the expected time left.
                            Args:
                                file_name (str): the file_name of the saved RNN

                            Returns:
                                rnn (RNN): the loaded rnn
                            """
    with open(f"data/saved_rnns/{file_name}.pickle", 'rb') as handle:
        return pickle.load(handle)

def test_rnn(rnn, en_data_file_name="tokenized_test_en", fr_data_file_name="tokenized_test_fr", criterion=bleu_score, batch_size=256, max_seq_length=60):
    """Computes the error of the rnn on the dataset

                            Args:
                                rnn (RNN): the recurrent neural network the english data is passed through
                                en_data_file_name (str): the english data input
                                fr_data_file_name (str):the french expected data output
                                criterion (function): the loss function
                                batch_size (int): the size of the batches used by the DataLoader
                                max_seq_length (int): maximum length at which input sentences are cut off

                            Returns:
                                error (float): the average error

                            """
    rnn.encoder.embedding.to(device)
    rnn.encoder.lstm.to(device)
    rnn.decoder.embedding.to(device)
    rnn.decoder.lstm.to(device)
    rnn.encoder.dropout = nn.Dropout(0)
    rnn.decoder.out.to(device)

    dataset = Test_dataset(f"data/{en_data_file_name}.csv", f"data/{fr_data_file_name}.csv",
                        "data/en_lang_abridged_90.pickle",
                        "data/fr_lang_abridged_90.pickle", sequence_length=max_seq_length)

    test_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    error = rnn.test(test_dataloader, criterion=criterion)
    return error

# Code used to train the rnns during hyperparameter optimisation
"""for learning_rate in [0.01, 0.001]:
    for hidden_dim in [200, 100, 50]:
        print("learning_rate:", learning_rate, "hidden_dim:", hidden_dim)
        rnn = create_rnn(n_epochs=30, batch_size = 256, hidden_dim=100, plot_name=f"hidden_dim_{hidden_dim}_lr_{learning_rate}_plot")
        save_rnn(rnn, f"hidden_dim_{hidden_dim}_lr_{learning_rate}")"""

# code used to extract saved training data to fill the tables
"""plot_name = "hidden_dim_50_lr_0.001_plot"
with open(f'data/{plot_name}_data.pickle', 'rb') as handle:
    data = pickle.load(handle)
    print("last training error:", data["plot_losses"][len(data["plot_losses"]) - 1])
    print("last validation errors:", data["plot_validation_losses"][len(data["plot_validation_losses"]) - 1])
    print("least validation error:", min(data["plot_validation_losses"]))
    print("at:", 1 + data["plot_validation_losses"].index(min(data["plot_validation_losses"])))
    print("train time:", data["time"]/3600)"""


# code used to calculate the bleu score on test data for a saved rnn
"""rnn = load_rnn("hidden_dim_100_lr_0.01")
bleu_score_output = test_rnn(rnn, criterion=my_bleu_score, batch_size=512)
print(bleu_score_output)
with open(f'data/bleu_score_output_data.pickle', 'wb') as handle:
    pickle.dump(bleu_score_output, handle)"""







