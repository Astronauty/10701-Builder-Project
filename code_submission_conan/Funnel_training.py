import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import math
import copy
import spacy
import pathlib
from pathlib import Path
import csv
import numpy as np
from torchtext.data.metrics import bleu_score
from Funnel_Transformer import *
from data_loader import *
from data_loader_tokens import Test_dataset
import datetime

from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here

# initialized the datasets for validation and training
train_ds = Test_dataset("data/tokenized_train_en_abridge.csv","data/tokenized_train_fr_abridge.csv","data/en_lang_80.pickle", "data/fr_lang_80.pickle", sequence_length=100)
val_ds = Test_dataset("data/tokenized_val_en_abridge.csv","data/tokenized_val_fr_abridge.csv","data/en_lang_80.pickle", "data/fr_lang_80.pickle", sequence_length=100)

# get the device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# initialize transformer parameters
# acquire the vocab sizes
src_vocab_size = train_ds.en_lang.n_words
tgt_vocab_size = train_ds.fr_lang.n_words
# initialize the model parameters
d_model = 512
num_heads = 8
d_ff = 512
max_seq_length = 60
encoder_blocks = 3
# make sure the encoder block is valid
assert((max_seq_length%(2**(encoder_blocks-1))==0)),"This shape is not compatible"
assert((max_seq_length/(2**(encoder_blocks-1))!=0)),"This shape is not compatible"
decoder_blocks = 8
dropout = 0.1

#initialize model
f_transformer = Funnel_Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, decoder_blocks, encoder_blocks, d_ff, max_seq_length, dropout)
f_transformer = f_transformer.to(device)

# initialize loss
loss_criterion = nn.CrossEntropyLoss(ignore_index=0)
loss_criterion.to(device)

#initialize dataloaders
dataloader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=6)
dataloader_val = DataLoader(val_ds, batch_size=256, shuffle=True, num_workers=6)

#initialize optimizer
optimizer = optim.Adam(f_transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.1, patience=10, threshold=0.0001, threshold_mode='abs')

#initialize tensorboard
log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter('runs/'+log_dir)
total_loss = 0.0
avg_loss = 0.0
# start training
for small_epoch in range(60):
    f_transformer.train()
    small_epoch_count = 0
    for i,it in enumerate(tqdm(dataloader)):
        src_data, tgt_data = it
        src_data = src_data.to(device)
        src_data = torch.squeeze(src_data)
        tgt_data = tgt_data.to(device)
        tgt_data = torch.squeeze(tgt_data)
        optimizer.zero_grad()
        output= f_transformer(src_data, tgt_data[:, :-1])
        loss = loss_criterion(output[0].contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        del loss
        del output
    
    with torch.no_grad():
        f_transformer.eval()
        total_loss = 0.0
        for i,it in enumerate(tqdm(dataloader_val)):
            src_data, tgt_data = it
            src_data = src_data.to(device)
            tgt_data = tgt_data.to(device)
            # tgt shifted
            output= f_transformer(src_data, tgt_data[:, :-1])
            loss = loss_criterion(output[0].contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
            total_loss += loss.item()
        avg_loss = total_loss / (i + 1)
    scheduler.step(avg_loss)
    writer.add_scalar('training loss',
                    avg_loss,
                        small_epoch)
    print(f"Epoch: {small_epoch+1}, Epoch Validation Loss: {avg_loss}")
checkpoint = { 
    'epoch': small_epoch,
    'model': f_transformer.state_dict(),
    'optimizer': optimizer.state_dict(),
    'lr_sched': scheduler}
torch.save(checkpoint, 'model_ckpt/'+log_dir+'.pth')
