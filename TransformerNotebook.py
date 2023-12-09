from DatasetSize import DatasetSize

from torch.nn import CrossEntropyLoss
from transformer import TransformerMT
from torch import optim
from TrainEval import TrainEval
from DataLoaderProvider import DataLoaderProvider
import pickle
from torchtext.data import get_tokenizer
from data_loader import Langs

en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer(tokenizer='spacy', language='fr_core_news_sm')

dataset_size = DatasetSize.MEDIUM

max_sequence_length = 50
dataloader_provider = DataLoaderProvider(dataset_size=dataset_size,
                                         batch_size=256,
                                         max_sequence_length=max_sequence_length)

en_vocab_path = f"data/{dataset_size.value}/vocabs/en_vocab.pkl"
fr_vocab_path = f"data/{dataset_size.value}/vocabs/fr_vocab.pkl"

with open(en_vocab_path, 'rb') as file:
    en_vocab: Langs = pickle.load(file)
with open(fr_vocab_path, 'rb') as file:
    fr_vocab: Langs = pickle.load(file)

model = transformer_mt = TransformerMT(
    source_vocabulary_size=en_vocab.n_words,
    target_vocabulary_size=fr_vocab.n_words,
    embedding_size=512,
    max_num_embeddings=max_sequence_length,
    num_attention_heads=8,
    num_encoder_layers=4,
    num_decoder_layers=4,
    linear_layer_size=1024,
    dropout=0.1,
    activation='relu',
    layer_norm_eps=1e-5,
    batch_first=True,
    norm_first=False,
    bias=True
)

train_eval = TrainEval(
    dataloader_provider=dataloader_provider,
    num_epochs=50,
    optimizer=optim.Adam(transformer_mt.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9),
    loss_function=CrossEntropyLoss(ignore_index=0),
    model=model,
    model_shortname='transformer',
    disambiguator='probe3'
)

train_eval.execute()