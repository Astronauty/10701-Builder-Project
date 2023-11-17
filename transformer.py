import math

from torch import Tensor
from torch.nn import Module, Embedding, Dropout, Transformer, Linear
import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
        positional_encodings = torch.zeros(max_num_embeddings, embedding_size)

        # create column of position indices corresponding to the max_num_embeddings
        positions = torch.arange(start=0, end=max_num_embeddings).unsqueeze(1)

        # compute the denominator term used in both the sin-based and cos-based equations
        shared_denominator = 10000 ** ((2 * positions) / embedding_size)

        # set the even positional encodings
        positional_encodings[:, 0::2] = torch.sin(positions * shared_denominator)

        # set the odd positional encodings
        positional_encodings[:, 1::2] = torch.cos(positions * shared_denominator)

        # add a batch dimension
        positional_encodings.unsqueeze(0)

        # add positional encodings to buffer so that they are frozen and not affected by backprop
        self.register_buffer('positional_encodings', positional_encodings)

        self.dropout = Dropout(p=0.1)

    def forward(self, token_embeddings):
        # overlay positional encodings onto token embeddings (via addition)
        position_encoded_token_embeddings = token_embeddings + self.positional_encodings[:, :self.embedding_size]

        # apply dropout before returning
        return self.dropout(position_encoded_token_embeddings)


class TransformerMT(Module):

    def __init__(self,
                 source_vocabulary_size: int,
                 target_vocabulary_size: int,
                 embedding_size: int,        # 512
                 max_num_embeddings: int,
                 num_attention_heads: int,   # 8
                 num_encoder_layers: int,    # 6
                 num_decoder_layers: int,    # 6
                 linear_layer_size: int,     # 2048
                 dropout: float,             # 0.1
                 activation: str,            # 'relu'
                 layer_norm_eps: float,        # 1e-5
                 batch_first: bool,          # True
                 norm_first: bool,           # False
                 bias: bool,                 # True
                 # src_target_length
                 ):
        super().__init__()

        self.source_token_embedder = TokenEmbedder(vocabulary_size=source_vocabulary_size,
                                                   embedding_size=embedding_size)

        self.target_token_embedder = TokenEmbedder(vocabulary_size=target_vocabulary_size,
                                                   embedding_size=embedding_size)

        self.positional_encoder = PositionalEncoder(max_num_embeddings=max_num_embeddings,
                                                    embedding_size=embedding_size)

        self.transformer = Transformer(d_model=embedding_size,
                                       nhead=num_attention_heads,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=linear_layer_size,
                                       dropout=dropout,
                                       activation=activation,
                                       layer_norm_eps=layer_norm_eps,
                                       batch_first=batch_first,
                                       norm_first=norm_first,
                                       bias=bias)

        self.final_linear_layer = Linear(embedding_size, target_vocabulary_size)

    def generate_mask(self, src, tgt):
        src_seq_len = src.shape[0]
        tgt_seq_len = tgt.shape[0]

        tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

        PAD = 2
        src_padding_mask = (src == PAD).transpose(0, 1)
        tgt_padding_mask = (tgt == PAD).transpose(0, 1)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self,
                src: Tensor,
                tgt: Tensor,
                src_mask,
                tgt_mask,
                src_key_padding_mask,
                tgt_key_padding_mask):
        # token embeddings
        source_token_embedding = self.source_token_embedder(src)

        target_token_embedding = self.target_token_embedder(tgt)

        # apply positional encodings
        position_encoded_source_token_embedding = self.positional_encoder(source_token_embedding)

        position_encoded_target_token_embedding = self.positional_encoder(target_token_embedding)

        transformer_out = self.transformer(src=position_encoded_source_token_embedding,
                                           tgt=position_encoded_target_token_embedding,
                                           src_mask=src_mask,
                                           tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_key_padding_mask,
                                           tgt_key_padding_mask=tgt_key_padding_mask)

        target_vocabulary_logits = self.final_linear_layer(transformer_out)

        return target_vocabulary_logits

