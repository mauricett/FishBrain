import time

import numpy as np
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from metrics.accuracy import Tester
from data.processors import process_sample, scorer
from data.tokenizer import Tokenizer
from model.conv_nhwc_v0.model import ConvTransformer
import matplotlib.pyplot as plt


# Neural net parameters
BATCHSIZE = 1024
D_EMB = 128
N_LAYERS = 4
N_HEADS = 1
device = 'cuda'
torch.set_float32_matmul_precision('high')

# Data / DataLoader
RESUME = True
LOAD_PATH = 'model/conv_nhwc_v0/'
SAVE_PATH = 'model/'
DATASET = '../FishData/lichess_sf.py'
NUM_WORKERS = 4

# Training loop
N_CHECKPOINT = 10000
N_EPOCHS = 100
MIXED_PRECISION = True
GRADIENT_NORM = 5

tokenizer = Tokenizer()
tester = Tester(batchsize=BATCHSIZE, tokenizer=tokenizer)

dataset = load_dataset(path=DATASET,
                       split='train',
                       streaming=True,
                       trust_remote_code=True)
dataset = dataset.map(function=process_sample, 
                      fn_kwargs={'tokenizer': tokenizer, 'scorer': scorer})

model = ConvTransformer(D_EMB, N_LAYERS, N_HEADS)
model = model.to(device)
model = torch.compile(model)

model_dict = {'acc': np.zeros((1, 62, 100)),
              'steps': [0],
              'loss': [],
              'batchsize': [BATCHSIZE]}

optimizer = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
bce_loss = nn.BCEWithLogitsLoss()

if RESUME:
    model.load_state_dict(torch.load(LOAD_PATH  + 'fishweights.pt'))
    model_dict = torch.load(LOAD_PATH  + 'model_dict.pt')
    optimizer.load_state_dict(torch.load(LOAD_PATH  + 'optimizer.pt'))

n_steps = model_dict['steps'][-1]
timer = time.perf_counter()

for epoch in range(N_EPOCHS):
    print('Epoch %i' % epoch)

    dataset = dataset.shuffle()
    dataloader = DataLoader(dataset, 
                        batch_size=BATCHSIZE,
                        num_workers=NUM_WORKERS)

    for batch in dataloader:
        optimizer.zero_grad()

        # We manually drop the last batch to avoid OOM problem with compiler.
        if len(batch['scores']) < BATCHSIZE:
            break

        with torch.autocast(device_type='cuda', enabled=MIXED_PRECISION, dtype=torch.bfloat16):
            x = model(batch['fens'].cuda(), batch['moves'].cuda())
            scores = batch['scores'].to(device)
            loss = bce_loss(x, scores)

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_NORM)
        optimizer.step()

        n_steps += 1

        model_dict['loss'].append(loss.item())

        if not (n_steps % N_CHECKPOINT):
            speed = (N_CHECKPOINT * BATCHSIZE) / (time.perf_counter() - timer)

            accuracy = tester(model)
            model_dict['acc'] = np.concatenate([model_dict['acc'], accuracy])
            model_dict['steps'].append(n_steps)
            model_dict['batchsize'].append(BATCHSIZE)
            
            print('%.1f accuracy, %i positions / s' % \
                    (model_dict['acc'][-1].mean() * 100, speed))

            torch.save(model.state_dict(), SAVE_PATH + 'fishweights.pt')
            torch.save(optimizer.state_dict(), SAVE_PATH + 'optimizer.pt')
            torch.save(model_dict, SAVE_PATH + 'model_dict.pt')

            timer = time.perf_counter()