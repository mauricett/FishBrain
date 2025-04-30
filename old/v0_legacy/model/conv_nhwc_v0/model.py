#%%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Embed(nn.Module):
    def __init__(self, D_EMB, SCALE):
        super().__init__()
        self.D_EMB = D_EMB
        self.rank_emb = nn.Parameter(torch.randn(8, 1, self.D_EMB) * SCALE)
        self.file_emb = nn.Parameter(torch.randn(1, 8, self.D_EMB) * SCALE)
        self.fen_emb = nn.Parameter(torch.randn(17, self.D_EMB) * SCALE)
        self.move_emb = nn.Parameter(torch.randn(2, self.D_EMB) * SCALE)
        self.abs_emb = nn.Parameter(torch.randn(71, self.D_EMB) * SCALE)

    def forward(self, fen, move):
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

class MHA(nn.Module):
    def __init__(self, D_EMB, SCALE, N_heads):
        super().__init__()
        self.H = N_heads
        self.D_heads = D_EMB // N_heads
        self.d_emb = torch.tensor(D_EMB).cuda()
        qkv_size = (3, N_heads, D_EMB, D_EMB // N_heads)
        self.qkv = nn.Parameter(torch.randn(qkv_size) * SCALE)

    def forward(self, x):
        B, T, D = x.shape
        x = torch.einsum('btd, nhdD -> nbhtD', x, self.qkv)
        q, k, v = x[0], x[1], x[2]
        attn = q @ k.transpose(-1, -2) / torch.sqrt(self.d_emb // self.H)
        attn = F.softmax(attn, -1)
        x = torch.einsum('bhtT, bhTd -> bthd', attn, v)
        return x.reshape(B, T, D)

class LayerNorm(nn.Module):
    def __init__(self, D):
        super().__init__()
        #self.W = nn.Parameter(torch.ones(1, D, 1))
        #self.b = nn.Parameter(torch.zeros(1, D, 1))
    def forward(self, x):
        s, m = torch.std_mean(x, dim=-2, keepdim=True)
        #return x / torch.sqrt(torch.var(x, dim=-2, keepdim=True) + 1e-5)
        return (x - m) / (s + 1e-5)# * self.W + self.b

class ConvTransformerBlock(nn.Module):
    def __init__(self, D_EMB, SCALE, N_heads):
        super().__init__()
        self.conv1 = nn.Conv2d(D_EMB, D_EMB, kernel_size=(3,3), padding='same', bias=False)
        self.conv1 = self.conv1.to(memory_format=torch.channels_last)
        self.conv2 = nn.Conv2d(D_EMB, D_EMB, kernel_size=(3,3), padding='same', bias=False)
        self.conv2 = self.conv2.to(memory_format=torch.channels_last)
        self.attention = MHA(D_EMB, SCALE, N_heads)
        #self.gate1 = nn.Parameter(torch.tensor(0.))
        #self.gate2 = nn.Parameter(torch.tensor(0.))
        #self.ln1 = nn.LayerNorm(D_EMB)
        #self.ln2 = nn.LayerNorm(D_EMB)
        self.ln1 = LayerNorm(64)
        self.ln2 = LayerNorm(64)

    def forward(self, x):
        B, T, D = x.shape

        y = self.ln1(x)
        y = y.view(B, 8, 8, D).permute((0, 3, 1, 2))
        y = F.gelu(y)
        y = self.conv1(y)
        y = F.gelu(y)
        y = self.conv2(y).permute((0, 2, 3, 1)).view(B, T, D)

        #gate1 = torch.sigmoid(self.gate1)
        #y = self.attention(torch.sqrt(1-gate1)*x + torch.sqrt(gate1)*y)
       
        #gate2 = torch.sigmoid(self.gate2)
        #return torch.sqrt(1-gate2)*x + torch.sqrt(gate2)*y

        x = x + y

        #return 0.707 * (x + self.ln2(self.attention(self.ln3(x))))
        #return x + self.attention(self.ln2(x))
        y = self.attention(self.ln2(x))
        return x + y

class ConvTransformer(nn.Module):
    def __init__(self, D_EMB, N_layers, N_heads):
        super().__init__()
        SCALE = 0.02
        self.D_EMB = D_EMB
        self.H = N_heads

        self.embed = Embed(D_EMB, SCALE)
        self.ln_embed = nn.LayerNorm(D_EMB)
        self.embed_attn = MHA(D_EMB, SCALE, N_heads)
        self.ln_embed2 = LayerNorm(64)

        self.transformer = nn.ModuleList(
            [ConvTransformerBlock(D_EMB, SCALE * 0.64, self.H) for i in range(N_layers)])

        self.ln_out = LayerNorm(64)
        self.conv_out1 = nn.Conv2d(D_EMB, 2 * D_EMB, kernel_size=(3, 3), padding='same', bias=False)
        self.conv_out1 = self.conv_out1.to(memory_format=torch.channels_last)
        self.conv_out2 = nn.Conv2d(2 * D_EMB, 1, kernel_size=(8, 8), bias=False)
        self.conv_out2 = self.conv_out2.to(memory_format=torch.channels_last)

    def forward(self, fen, move):
        x = self.ln_embed(self.embed(fen, move))
        y = self.ln_embed2(self.embed_attn(x))
        x = 0.707 * (x[:, :64] + y[:, :64])
        B, T, D = x.shape
        
        for layer in self.transformer:
            x = layer(x)
            #plt.imshow(x[0].reshape(T, D).cpu().detach())
            #plt.show()

        x = self.ln_out(x)
        x = self.conv_out1(x.view(B, 8, 8, D).permute((0, 3, 1, 2)))
        x = F.gelu(x)
        x = self.conv_out2(x)
        return x.flatten()