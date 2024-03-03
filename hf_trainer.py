#%%
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
from model.conv_fast_v0.model import ConvTransformer


BATCHSIZE = 1024
N_CHECKPOINT = 2500
D_EMB = 128
N_LAYERS = 4
N_HEADS = 4
device = 'cuda'

tokenizer = Tokenizer()
tester = Tester(batchsize=BATCHSIZE, tokenizer=tokenizer)

dataset = load_dataset(path="mauricett/lichess_sf",
                       split="train",
                       streaming=True,
                       trust_remote_code=True)

dataset = dataset.map(function=process_sample, 
                      fn_kwargs={"tokenizer": tokenizer, "scorer": scorer})

dataloader = DataLoader(dataset, 
                        batch_size=BATCHSIZE,
                        num_workers=4)

#%%
model = ConvTransformer(D_EMB, N_LAYERS, N_HEADS)
model = model.to(device)
model = torch.compile(model)
model_dict = {'acc': np.zeros((1, 62, 100)),
              'steps': [0],
              'loss': [],
              'batchsize': [BATCHSIZE],
              'grad_norm': []}

optimizer = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
bce_loss = nn.BCEWithLogitsLoss()

max_norm = 3

#%%
n_steps = 0
n_epochs = 100
timer = time.perf_counter()

for epoch in range(n_epochs):
    print("Epoch %i" % epoch)

    dataset = dataset.shuffle()

    for batch in dataloader:
        optimizer.zero_grad()

        x = model(batch['fens'].cuda(), batch['moves'].cuda())
        scores = batch['scores'].float().to(device)
        loss = bce_loss(x, scores)
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        model_dict['grad_norm'].append(float(grad_norm.cpu()))

        optimizer.step()
        n_steps += 1

        model_dict['loss'].append(loss.item())

        if not (n_steps % N_CHECKPOINT):
            speed = (N_CHECKPOINT * BATCHSIZE) / (time.perf_counter() - timer)

            accuracy = tester(model)
            model_dict['acc'] = np.concatenate([model_dict['acc'], accuracy])
            model_dict['steps'].append(n_steps)
            model_dict['batchsize'].append(BATCHSIZE)

            print("%.1f accuracy, %i positions / s" % \
                    (model_dict['acc'][-1].mean() * 100, speed))

            torch.save(model.state_dict(), 'model/fishweights.pt')
            torch.save(optimizer.state_dict(), "model/optimizer.pt")
            torch.save(model_dict, "model/model_dict.pt")

            timer = time.perf_counter()