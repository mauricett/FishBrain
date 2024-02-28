import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, weight_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(weight_shape))

    def forward(self, x, dim):
        denom = torch.pow(x, 2).mean(dim, keepdim=True)
        x = x / torch.sqrt(denom + self.eps)
        return self.weights * x


class MHA(nn.Module):
    def __init__(self, D_EMB, SCALE, N_heads):
        super().__init__()
        self.H = N_heads
        self.d_emb = torch.tensor(D_EMB)

        self.norm = RMSNorm(weight_shape=(1,))

        qkv_shape = (3, N_heads, D_EMB, D_EMB // N_heads)
        self.qkv = nn.Parameter(torch.randn(qkv_shape) * SCALE)

    def forward(self, x):
        B, D, T = x.shape

        x = F.leaky_relu(x, negative_slope=0.2)
        x = torch.einsum('bdt, nhdD -> nbhtD', x, self.qkv)
        x = self.norm(x, dim=-2)

        q, k, v = x[0], x[1], x[2]
        attn = q @ k.transpose(-1, -2) / torch.sqrt(self.d_emb // self.H)
        attn = F.softmax(attn, -1)
        x = torch.einsum('bhtT, bhTD -> bhDt', attn, v)

        return x.reshape(B, D, T)


class ConvBlock(nn.Module):
    def __init__(self, D_EMB, SCALE):
        super().__init__()
        
        self.conv = nn.Conv2d(D_EMB, D_EMB, kernel_size=(3,3), padding='same')
        self.conv.weight.data *= SCALE
        self.conv.bias.data *= 0
        
        self.conv2 = nn.Conv2d(D_EMB, D_EMB, kernel_size=(3,3), padding='same')
        self.conv2.weight.data *= SCALE
        self.conv2.bias.data *= 0

        self.norm = RMSNorm(weight_shape=(1,))

    def forward(self, x):
        B, D, T = x.shape
        x = x.reshape(B, D, 8, 8)

        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv(x)

        x = self.norm(x, dim=(-1, -2))

        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv2(x)

        return x.reshape(B, D, T)


class ConvTransformerBlock(nn.Module):
    def __init__(self, D_EMB, SCALE, N_heads):
        super().__init__()
        self.attention = MHA(D_EMB, SCALE, N_heads)
        self.conv = ConvBlock(D_EMB, SCALE)

    def forward(self, x):
        B, D, T = x.shape

        y = self.conv(x)
        x = x + y

        y = self.attention(x)
        x = x + y

        return x


class Net(nn.Module):
    def __init__(self, D_EMB, N_layers, N_heads):
        super().__init__()
        SCALE = 0.01
        self.D_EMB = D_EMB
        self.H = N_heads

        self.rank_emb = nn.Parameter(torch.randn(8, 1, self.D_EMB) * SCALE)
        self.file_emb = nn.Parameter(torch.randn(1, 8, self.D_EMB) * SCALE)
        self.fen_emb = nn.Parameter(torch.randn(17, self.D_EMB) * SCALE)
        self.move_emb = nn.Parameter(torch.randn(2, self.D_EMB) * SCALE)
        self.abs_emb = nn.Parameter(torch.randn(71, self.D_EMB) * SCALE)

        self.emb_attention = MHA(D_EMB, SCALE, self.H)
        self.layernorm = nn.LayerNorm(71)

        self.transformer = nn.ModuleList(
            [ConvTransformerBlock(D_EMB, SCALE * 0.64, self.H) for i in range(N_layers)])

        self.norm = RMSNorm(weight_shape=(1,))
        
        self.conv_out = nn.Conv2d(D_EMB, 2 * D_EMB, kernel_size=(3, 3), padding='same')
        self.conv_out.weight.data *= SCALE
        self.conv_out.bias.data *= 0
        
        self.conv_out2 = nn.Conv2d(2 * D_EMB, 1, kernel_size=(8, 8))
        self.conv_out2.weight.data *= SCALE
        self.conv_out2.bias.data *= 0 

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

        x = self.layernorm(x)
        y = self.emb_attention(x)    
        x = x[:, :, :64] + y[:, :, :64]

        for layer in self.transformer:
            x = layer(x)
        
        x = x.view(B, D, 8, 8)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_out(x)
        
        x = self.norm(x, dim=(-1, -2))

        x = F.leaky_relu(x, negative_slope=0.2)
        x = self.conv_out2(x)

        return x.flatten()