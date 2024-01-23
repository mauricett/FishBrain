#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from data.dataset import LichessData, worker_init_fn
import matplotlib.pyplot as plt

#%%
lichess_data = LichessData(min_elo=0, resume=False)

# !!! Use num_workers <= 5 to prevent being rate-limited by Lichess !!!
dataloader = DataLoader(lichess_data, batch_size=256, 
                        num_workers=4, worker_init_fn=worker_init_fn)



#%%
class ChessTransformer(nn.Module):
    def __init__(self, D_EMB, N_layers):
        super().__init__()
        SCALE = 0.02
        self.D_EMB = D_EMB

        self.rank_emb = nn.Parameter(torch.randn(8, 1, self.D_EMB) * SCALE)
        self.file_emb = nn.Parameter(torch.randn(1, 8, self.D_EMB) * SCALE)
        self.fen_emb = nn.Parameter(torch.randn(17, self.D_EMB) * SCALE)
        self.move_emb = nn.Parameter(torch.randn(2, self.D_EMB) * SCALE)
        self.layernorm_emb = nn.LayerNorm(self.D_EMB)

        self.transformer = nn.ModuleList(
            [TransformerBlock(D_EMB, SCALE) for i in range(N_layers)])

        self.linear_out = nn.Linear(2 * self.D_EMB, 1)
        self.linear_out.bias.data *= 0.

    def forward(self, fen, move):
        x = self.embed(fen, move)
        x = self.layernorm_emb(x)

        for layer in self.transformer:
            x = layer(x)
        
        x = x[:, 69:].reshape(x.shape[0], 2 * self.D_EMB)
        x = self.linear_out(x)
        x = torch.sigmoid(x)
        return x

    def embed(self, fen, move):
        pos_emb = (self.rank_emb + self.file_emb).view(64, self.D_EMB)
        move = pos_emb[move] + self.move_emb
        move *= 0.58 

        fen = self.fen_emb[fen]

        pieces = fen[:, :64]
        colors = fen[:, 64:128]
        flags = fen[:, 128:]

        board = 0.5 * (pieces + colors + pos_emb)

        return torch.cat([board, flags, move], dim=1)


class TransformerBlock(nn.Module):
    def __init__(self, D_EMB, SCALE):
        super().__init__()
        self.d_emb = torch.tensor(D_EMB)
        qkv_shape = (3, D_EMB, D_EMB)
        self.qkv = nn.Parameter(torch.randn(qkv_shape) * SCALE)
        self.ln_q = nn.LayerNorm(D_EMB)
        self.ln_k = nn.LayerNorm(D_EMB)
        self.layernorm_attn = nn.LayerNorm(D_EMB)

        self.lin1 = nn.Linear(D_EMB, 2 * D_EMB, bias=False)
        self.lin2 = nn.Linear(2 * D_EMB, D_EMB, bias=False)
        self.layernorm_ff = nn.LayerNorm(D_EMB)

    def forward(self, x):
        y = torch.einsum('btd, ndD -> nbtD', x, self.qkv)
        q, k, v = y[0], y[1], y[2]
        q = self.ln_q(q)
        k = self.ln_k(k)
        y = q @ k.transpose(-1, -2) / torch.sqrt(self.d_emb)
        y = F.softmax(y, dim=-1)
        y = y @ v
        y = self.layernorm_attn(y)
        x = 0.7 * (x + y)

        y = self.lin1(x)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = self.lin2(y)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = self.layernorm_ff(y)
        x = 0.7 * (x + y)
        return x

#%%
if __name__ == '__main__':
    model = ChessTransformer(D_EMB=1024, N_layers=8)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    #%%
    print("start")
    bce_loss = nn.BCELoss()

    loss_plot = []
    n = 0
    for batch in dataloader:
    
        fen = batch['fen'].cuda()
        move = batch['move'].cuda()
        score = batch['score'].cuda()

        optimizer.zero_grad()
        x = model(fen, move).flatten()

        #loss = (x - score)**2
        #loss = loss.mean()

        loss = bce_loss(x, score)
        loss.backward()
        loss_plot.append(loss.item())

        optimizer.step()

        if not (n % 50):
            plt.plot(loss_plot[30:])
            plt.show()
            torch.save(model.state_dict(), 'model/fishweights.pt')