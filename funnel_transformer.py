
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
from enum import Enum


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads # d_k dimensionality of the word embeddings in each head
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_b = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None): 
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(self.d_k) # Q^T*K and normalize by sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9) # convert zeros to -1e9 to force softmax to zero
        attention = nn.Softmax(dim=-1)(scores)
        return torch.matmul(attention, V)
    
    def split_heads(self, x):
        # x is tensor of all sequences and words
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        batch_size, _, seq_len, d_model = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
    
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
    
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super(Transformer, self).__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))

        enc_output = src_embedded
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)

        dec_output = tgt_embedded
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)

        output = self.fc(dec_output)
        return output
# class FunnelTransformer(nn.Module):
#     def __init__(self, num_tokens, emb_size, hidden_size, num_layers, num_heads, dropout):
#         super().__init__()
#         self.num_tokens = num_tokens
#         self.emb_size = emb_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.num_heads = num_heads
#         self.dropout = dropout
        
#         self.token_embedding = nn.Embedding(num_tokens, emb_size)
#         self.positional_embedding = nn.Parameter(torch.zeros(1, num_tokens, emb_size))
#         self.dropout_layer = nn.Dropout(dropout)
        
#         self.encoder_layers = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         self.decoder_layers = nn.ModuleList([
#             nn.TransformerDecoderLayer(d_model=emb_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
#             for _ in range(num_layers)
#         ])
        
#         self.encoder_norm = nn.LayerNorm(emb_size)
#         self.decoder_norm = nn.LayerNorm(emb_size)
        
#         self.output_layer = nn.Linear(emb_size, num_tokens)
        
#     def forward(self, input_tokens, target_tokens):
#         input_embeddings = self.token_embedding(input_tokens) + self.positional_embedding
#         target_embeddings = self.token_embedding(target_tokens) + self.positional_embedding
        
#         input_embeddings = self.dropout_layer(input_embeddings)
#         target_embeddings = self.dropout_layer(target_embeddings)
        
#         for i in range(self.num_layers):
#             input_embeddings = self.encoder_layers[i](input_embeddings)
#             input_embeddings = self.encoder_norm(input_embeddings)
            
#             target_embeddings = self.decoder_layers[i](target_embeddings, input_embeddings)
#             target_embeddings = self.decoder_norm(target_embeddings)
        
#         output_logits = self.output_layer(target_embeddings)
#         return output_logits
