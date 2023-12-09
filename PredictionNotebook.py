from Prediction import Prediction
import torch
from DatasetSize import DatasetSize
from transformer import TransformerMT
from DataLoaderProvider import DataLoaderProvider
import pickle
from torchtext.data import get_tokenizer
from data_loader import Langs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

en_tokenizer = get_tokenizer(tokenizer='spacy', language='en_core_web_sm')
fr_tokenizer = get_tokenizer(tokenizer='spacy', language='fr_core_news_sm')

dataset_size = DatasetSize.MEDIUM

max_sequence_length = 50
dataloader_provider = DataLoaderProvider(dataset_size=dataset_size,
                                         batch_size=128,
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

checkpoint = torch.load('checkpoint/m-transformer_e-9_l-1.7169042542157575_d-probe3.pt')

model.load_state_dict(checkpoint['model_state_dict'])

model = model.to(device)

prediction = Prediction(model=model, test_dataloader=dataloader_provider.get_test_dataloader(), vocabulary=fr_vocab)

bleu_score = prediction.get_bleu_score()

print(f"FINAL BLEU SCORE: {bleu_score}")