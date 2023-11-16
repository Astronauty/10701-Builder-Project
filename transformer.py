import math

from torch import Tensor
from torch.nn import Module, Embedding, Dropout
import torch


class TokenEmbedder(Module):

    def __init__(self, vocabulary_size: int, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

        self.embedding = Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size)

    def forward(self, token_ids: Tensor):
        token_embeddings = self.embedding(token_ids)

        token_embeddings = token_embeddings * math.sqrt(self.embedding_size)

        return token_embeddings


class PositionalEncoder(Module):

    def __init__(self, max_num_embeddings: int, embedding_size: int):
        super().__init__()
        self.embedding_size = embedding_size

        # initialize a tensor that will eventually store the positional encodings
        self.positional_encodings = torch.zeros(max_num_embeddings, embedding_size)

        # create column of position indices corresponding to the max_num_embeddings
        positions = torch.arange(start=0, end=max_num_embeddings).unsqueeze(1)

        # compute the denominator term used in both the sin-based and cos-based equations
        shared_denominator = 10000 ** ((2 * positions) / embedding_size)

        # set the even positional encodings
        self.positional_encodings[:, 0::2] = torch.sin(positions * shared_denominator)

        # set the odd positional encodings
        self.positional_encodings[:, 1::2] = torch.cos(positions * shared_denominator)

        # add a batch dimension
        self.positional_encodings.unsqueeze(0)

        # add positional encodings to buffer so that they are frozen and not affected by backprop
        self.register_buffer('positional_encodings', self.positional_encodings)

        self.dropout = Dropout(p=0.1)

    def forward(self, token_embeddings):
        # overlay positional encodings onto token embeddings (via addition)
        position_encoded_token_embeddings = token_embeddings + self.positional_encodings[:, :self.embedding_size]

        # apply dropout before returning
        return self.dropout(position_encoded_token_embeddings)










# class TransformerMT(Module):
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # init embedding tensor
#         self.token_embedder = TokenEmbedder(emb_size=1, vocab_size=1)     # todo set this size (not 1)
#
#         # init position encoding tensor
#         self.positional_encoder = PositionalEncoder(emb_size=1, dropout=1, maxlen=1)    # todo
#
#         # init transformer
#         self.transformer = torch.nn.Transformer(d_model=emb_size,
#                                        nhead=nhead,
#                                        num_encoder_layers=num_encoder_layers,
#                                        num_decoder_layers=num_decoder_layers,
#                                        dim_feedforward=dim_feedforward,
#                                        dropout=dropout)
#
#     def forward(self, input):
#         # get dense embeddings tensor given input sparse vocab index tensor
#
#         # apply positional encoding transformation
#
#         # apply transformer
#         pass
#
#     # TODO: add other function(s) for usability during inference/translation
