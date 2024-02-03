import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        self.abs_emb = nn.Parameter(torch.randn(71, self.D_EMB) * SCALE)

        self.transformer = nn.ModuleList(
            [TransformerBlock(D_EMB, SCALE) for i in range(N_layers)])

        self.layernorm_out = nn.LayerNorm(2 * self.D_EMB)
        self.linear_out = nn.Linear(2 * self.D_EMB, 2 * self.D_EMB)
        self.linear_out.bias.data *= 0.
        self.linear_out2 = nn.Linear(2 * self.D_EMB, 1)
        self.linear_out2.bias.data *= 0.

    def forward(self, fen, move):
        x = self.embed(fen, move)
        x = self.layernorm_emb(x)

        for layer in self.transformer:
            x = layer(x)

        x = x[:, 69:].reshape(x.shape[0], 2 * self.D_EMB)
        x = self.layernorm_out(x)
        x = self.linear_out(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.linear_out2(x)
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
        x = torch.cat([board, flags, move], dim=1)
        x += self.abs_emb

        return x


class TransformerBlock(nn.Module):
    def __init__(self, D_EMB, SCALE):
        super().__init__()

        """ ignore these, I forgot to delete these """
        """ -------------------------------------- """
        self.d_emb = torch.tensor(D_EMB)
        self.ln_q = nn.LayerNorm(D_EMB)
        self.ln_k = nn.LayerNorm(D_EMB)
        self.prenorm1 = nn.LayerNorm(D_EMB)
        self.postnorm1 = nn.LayerNorm(D_EMB)

        self.lin1 = nn.Linear(D_EMB, 2 * D_EMB)
        self.lin2 = nn.Linear(2 * D_EMB, D_EMB)
        self.prenorm2 = nn.LayerNorm(D_EMB)
        self.postnorm2 = nn.LayerNorm(D_EMB)
        """ -------------------------------------- """

        self.H = 8
        qkv_shape = (3, self.H, D_EMB // self.H, D_EMB // self.H)
        self.qkv = nn.Parameter(torch.randn(qkv_shape) * SCALE)
        
        ff1_shape = (self.H, D_EMB, D_EMB // self.H)
        self.ff1 = nn.Parameter(torch.randn(ff1_shape) * SCALE)
        ff2_shape = (self.H, D_EMB // self.H, D_EMB // self.H)
        self.ff2 = nn.Parameter(torch.randn(ff2_shape) * SCALE)

    def forward(self, x):
        B, T, D = x.shape
        x = x.view(B, T, self.H, D // self.H)

        y = torch.einsum('bthd, nhdD -> nbhtD', x, self.qkv)
        q, k, v = y[0], y[1], y[2]

        y = q @ k.transpose(-1, -2) / torch.sqrt(self.d_emb // self.H)
        y = F.softmax(y, dim=-1)
        y = torch.einsum('bhTt, bhtd -> bThd', y, v)
        x = x + y
        x = x.view(B, T, D)

        y = torch.einsum('btd, hdD -> bthD', x, self.ff1)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = torch.einsum('bthd, hdD -> bthD', y, self.ff2)
        y = F.leaky_relu(y, negative_slope=0.2)
        
        x = x + y.reshape(B, T, D)
        return x
