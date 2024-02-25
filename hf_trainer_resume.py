#%%
import time

import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from metrics.accuracy import Tester
from data.processors import process_sample, scorer
from data.tokenizer import Tokenizer
from model.conv_big_v0.model import ConvTransformer


BATCHSIZE = 1024
N_CHECKPOINT = 10000
D_EMB = 256
N_LAYERS = 8
N_HEADS = 8
device = 'cuda'


""" 
---------------------------------------------
load dataset, initialize tokenizer and tester
---------------------------------------------
"""
tokenizer = Tokenizer()
tester = Tester(batchsize=BATCHSIZE, tokenizer=tokenizer)

dataset = load_dataset(path="mauricett/lichess_sf",
                       split="train",
                       streaming=True)

dataset = dataset.map(function=process_sample, 
                      fn_kwargs={"tokenizer": tokenizer, "scorer": scorer})

dataloader = DataLoader(dataset, 
                        batch_size=BATCHSIZE,
                        num_workers=4)


"""
---------------------------------------------
load model checkpoint
---------------------------------------------
"""
model = ConvTransformer(D_EMB, N_LAYERS, N_HEADS)
model = nn.DataParallel(model)
model = model.to(device)
model.load_state_dict(torch.load("model/conv_big_v0/fishweights.pt"))
model_dict = torch.load("model/conv_big_v0/model_dict.pt")

scaler = torch.cuda.amp.GradScaler(init_scale=2**16, growth_factor=1.5, backoff_factor=0.66)
scaler.load_state_dict(torch.load("model/conv_big_v0/scaler_dict.pt"))

optimizer = optim.Adam(model.parameters(), lr=3e-4, amsgrad=True)
optimizer.load_state_dict(torch.load("model/conv_big_v0/optimizer.pt"))

bce_loss = nn.BCEWithLogitsLoss()
max_norm = 5

n_steps = model_dict['steps'][-1]


"""
---------------------------------------------
training loop
---------------------------------------------
"""
timer = time.perf_counter()
n_epochs = 100

for epoch in range(n_epochs):
    print("Epoch %i" % epoch)

    dataset = dataset.shuffle()

    for batch in dataloader:
        optimizer.zero_grad()

        with torch.autocast(device_type="cuda"):
            x = model(batch['fens'], batch['moves'])
            scores = batch['scores'].to(device)
            loss = bce_loss(x, scores)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        n_steps += 1

        model_dict['grad_norm'].append(float(grad_norm.cpu()))
        model_dict['loss'].append(loss.item())

        if not (n_steps % N_CHECKPOINT):
            speed = (N_CHECKPOINT * BATCHSIZE) / (time.perf_counter() - timer)
            
            plt.plot(model_dict['loss'][30:])
            plt.show()

            accuracy = tester(model)
            model_dict['acc'] = np.concatenate([model_dict['acc'], accuracy])
            model_dict['steps'].append(n_steps)
            model_dict['batchsize'].append(BATCHSIZE)
            
            print("%.1f accuracy, %i positions / s" % \
                    (model_dict['acc'][-1].mean() * 100, speed))

            torch.save(model.state_dict(), 'model/fishweights.pt')
            torch.save(optimizer.state_dict(), "model/optimizer.pt")
            torch.save(model_dict, "model/model_dict.pt")
            torch.save(scaler.state_dict(), "model/scaler_dict.pt")

            timer = time.perf_counter()