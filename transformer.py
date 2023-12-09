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
        positional_encodings = positional_encodings.unsqueeze(0)

        # add positional encodings to buffer so that they are frozen and not affected by backprop
        self.register_buffer('positional_encodings', positional_encodings)

        self.dropout = Dropout(p=0.1)

    def forward(self, token_embeddings):
        # overlay positional encodings onto token embeddings (via addition)
        position_encoded_token_embeddings = token_embeddings + self.positional_encodings[:, :token_embeddings.size(1)]

        # apply dropout before returning
        return self.dropout(position_encoded_token_embeddings)


class TransformerMT(Module):

    def __init__(self,
                 source_vocabulary_size: int,
                 target_vocabulary_size: int,
                 embedding_size: int,
                 max_num_embeddings: int,
                 num_attention_heads: int,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 linear_layer_size: int,
                 dropout: float,
                 activation: str,
                 layer_norm_eps: float,
                 batch_first: bool,
                 norm_first: bool,
                 bias: bool):
        super().__init__()

        self.source_token_embedder = TokenEmbedder(vocabulary_size=source_vocabulary_size,
                                                   embedding_size=embedding_size)

        self.target_token_embedder = TokenEmbedder(vocabulary_size=target_vocabulary_size,
                                                   embedding_size=embedding_size)

        self.positional_encoder = PositionalEncoder(max_num_embeddings=max_num_embeddings,
                                                    embedding_size=embedding_size)

        self.transformer: Transformer = Transformer(d_model=embedding_size,
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

    def create_mask(self, src=None, tgt=None):
        PAD_IDX = 2
        src_mask, src_padding_mask = None, None
        if src is not None:
            src_seq_len = src.shape[1]
            src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)
            src_padding_mask = (src == PAD_IDX)

        tgt_mask, tgt_padding_mask = None, None
        if tgt is not None:
            tgt_seq_len = tgt.shape[1]
            tgt_mask = Transformer.generate_square_subsequent_mask(tgt_seq_len, device=DEVICE)
            tgt_padding_mask = (tgt == PAD_IDX)

        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self,
                src: Tensor,
                tgt: Tensor):
        src_mask, tgt_mask, _, _ = self.create_mask(src, tgt)

        # token embeddings
        source_token_embedding = self.source_token_embedder(src)

        target_token_embedding = self.target_token_embedder(tgt)

        # apply positional encodings
        position_encoded_source_token_embedding = self.positional_encoder(source_token_embedding)

        position_encoded_target_token_embedding = self.positional_encoder(target_token_embedding)

        transformer_out = self.transformer(src=position_encoded_source_token_embedding,
                                           tgt=position_encoded_target_token_embedding,
                                           src_mask=src_mask,
                                           tgt_mask=tgt_mask)

        target_vocabulary_logits = self.final_linear_layer(transformer_out)

        return target_vocabulary_logits

    def encode(self, src: Tensor):
        src_mask, _, _, _ = self.create_mask(src=src)

        source_token_embedding = self.source_token_embedder(src)

        position_encoded_source_token_embedding = self.positional_encoder(source_token_embedding)

        encoder_out = self.transformer.encoder(src=position_encoded_source_token_embedding,
                                               mask=src_mask)

        return encoder_out

    def decode(self, encoder_out: Tensor, tgt: Tensor):
        _, tgt_mask, _, _ = self.create_mask(tgt=tgt)

        target_token_embedding = self.target_token_embedder(tgt)

        position_encoded_target_token_embedding = self.positional_encoder(target_token_embedding)

        decoder_out = self.transformer.decoder(tgt=position_encoded_target_token_embedding,
                                               memory=encoder_out,
                                               tgt_mask=tgt_mask)

        target_vocabulary_logits = self.final_linear_layer(decoder_out)

        return target_vocabulary_logits