import torch
import torch.nn as nn
import torch.nn.functional as F


class MHA(nn.Module):
    def __init__(self, D_EMB, SCALE, N_heads):
        super().__init__()
        self.H = N_heads
        self.d_emb = torch.tensor(D_EMB)
        qkv_size = (3, N_heads, D_EMB, D_EMB // N_heads)
        self.qkv = nn.Parameter(torch.randn(qkv_size) * SCALE)

    def forward(self, x):
        B, D, T = x.shape
        x = torch.einsum('bdt, nhdD -> nbhtD', x, self.qkv)
        q, k, v = x[0], x[1], x[2]
        attn = q @ k.transpose(-1, -2) / torch.sqrt(self.d_emb // self.H)
        attn = F.softmax(attn, -1)
        x = torch.einsum('bhtT, bhTD -> bhDt', attn, v)
        return x.reshape(B, D, T)

class ConvTransformerBlock(nn.Module):
    def __init__(self, D_EMB, SCALE, N_heads):
        super().__init__()
        self.conv = nn.Conv2d(D_EMB, D_EMB, kernel_size=(3,3), padding='same')
        self.conv2 = nn.Conv2d(D_EMB, D_EMB, kernel_size=(3,3), padding='same')
        self.attention = MHA(D_EMB, SCALE, N_heads)
        self.ln = nn.LayerNorm(64)

    def forward(self, x):
        B, D, T = x.shape
        
        y = x.reshape(B, D, 8, 8)
        y = self.conv(y)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = self.conv2(y)
        y = F.leaky_relu(y, negative_slope=0.2)
        y = y.reshape(B, D, T)

        x = x + y

        y = self.ln(x)
        y = self.attention(y)

        return x + y

class ConvTransformer(nn.Module):
    def __init__(self, D_EMB, N_layers, N_heads):
        super().__init__()
        SCALE = 0.02
        self.D_EMB = D_EMB
        self.H = N_heads

        self.rank_emb = nn.Parameter(torch.randn(8, 1, self.D_EMB) * SCALE)
        self.file_emb = nn.Parameter(torch.randn(1, 8, self.D_EMB) * SCALE)
        self.fen_emb = nn.Parameter(torch.randn(17, self.D_EMB) * SCALE)
        self.move_emb = nn.Parameter(torch.randn(2, self.D_EMB) * SCALE)
        self.abs_emb = nn.Parameter(torch.randn(71, self.D_EMB) * SCALE)
        
        self.ln1 = nn.LayerNorm(71)
        self.emb_attention = MHA(D_EMB, SCALE, self.H)
        self.ln2 = nn.LayerNorm(64)

        self.transformer = nn.ModuleList(
            [ConvTransformerBlock(D_EMB, SCALE * 0.64, self.H) for i in range(N_layers)])

        self.ln3 = nn.LayerNorm(64)
        self.conv_out = nn.Conv2d(D_EMB, 2 * D_EMB, kernel_size=(3, 3), padding='same')
        self.conv_out2 = nn.Conv2d(2 * D_EMB, 1, kernel_size=(8, 8))

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
        x = x + self.abs_emb
        return x

    def forward(self, fen, move):
        x = self.embed(fen, move)
        x = x.transpose(-1, -2).contiguous()
        B, D, T = x.shape

        x = self.ln1(x)
        y = self.emb_attention(x)    
        x = x[:, :, :64] + y[:, :, :64]
        x = self.ln2(x)

        for layer in self.transformer:
            x = layer(x)

        x = self.ln3(x)
        x = x.view(B, D, 8, 8)
        x = self.conv_out(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_out2(x)
        return x.flatten()