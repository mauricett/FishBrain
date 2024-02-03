import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.dataset import LichessData, worker_init_fn
from model.fish_v0.model import ChessTransformer


model = ChessTransformer(D_EMB=512, N_layers=4)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=3e-4)

lichess_data = LichessData(min_elo=0, resume=False)
# !!! Use num_workers <= 5 to prevent being rate-limited by Lichess !!!
dataloader = DataLoader(lichess_data, batch_size=512, 
                        num_workers=4, worker_init_fn=worker_init_fn)

print("start")
bce_loss = nn.BCELoss()

loss_plot = []
n = 0
for batch in dataloader:

    fen = batch['fen'].cuda()
    move = batch['move'].cuda()
    score = batch['score'].cuda()

    optimizer.zero_grad()
    x = model(fen.long(), move.long()).flatten()

    loss = bce_loss(x, score)
    loss.backward()
    loss_plot.append(loss.item())

    optimizer.step()

    if not (n % 100):
        plt.plot(loss_plot[30:])
        plt.show()
        torch.save(model.state_dict(), 'model/fish_v0/fishweights.pt')
        torch.save(optimizer.state_dict(), "model/fish_v0/optimizer.pt")
        torch.save(loss_plot, "model/fish_v0/loss_plot.pt")
    
    n += 1
