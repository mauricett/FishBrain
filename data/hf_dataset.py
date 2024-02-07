#%%
import io
import random
import chess.pgn
from datasets import load_dataset
import chess
import torch
import torch.nn as nn
import numpy as np


class Tokenizer(nn.Module):
    def __init__(self):
        super(Tokenizer, self).__init__()
        # Define the mapping for pieces and colors
        self.piece_dict = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                           'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
        self.color_dict = {'r': 7, 'n': 7, 'b': 7, 'q': 7, 'k': 7, 'p': 7,
                           'R': 8, 'N': 8, 'B': 8, 'Q': 8, 'K': 8, 'P': 8}
        self.castle_dict = {'K': (0, 10), 'Q': (1, 11),
                            'k': (2, 12), 'q': (3, 13)}

        self.color_mirror = {'r': 8, 'n': 8, 'b': 8, 'q': 8, 'k': 8, 'p': 8,
                             'R': 7, 'N': 7, 'B': 7, 'Q': 7, 'K': 7, 'P': 7}
        self.castle_mirror = {'K': (0, 12), 'Q': (1, 13),
                              'k': (2, 10), 'q': (3, 11)}
                        
    def fen_fast2(self, fen, mirror):
        position, turn, castling, enpas = fen.split(' ')
        rows = position.split('/')

        if mirror:
            rows = rows[::-1]
            color_dict = self.color_mirror
            castle_dict = self.castle_mirror
        else:
            color_dict = self.color_dict
            castle_dict = self.castle_dict
        

        # Initialize tensors that hold indices for embedding tokens.
        #emb_idx = torch.zeros(133, dtype=torch.long)
        emb_idx = np.zeros(133, dtype=np.int64)
        emb_idx[64:128] = 9
        emb_idx[128:132] = 14
        emb_idx[-1] = 15

        # Iterate over the rows and columns and fill the tensors
        for i, row in enumerate(rows):
            col = 0
            for char in row:
                if char.isdigit():
                    # If the character is a digit, skip that many columns
                    col += int(char)
                else:
                    # Otherwise, fill the tensors with the piece and color values
                    idx = 8*i + col
                    emb_idx[idx] = self.piece_dict[char]
                    emb_idx[64 + idx] = color_dict[char]
                    col += 1
               
        # Each castling right gets a unique embedding token
        if castling != '-':
            for char in castling:
                i, embed_token = castle_dict[char]
                emb_idx[128 + i] = embed_token

        # En passent is a binary indicator for now (15=no, 16=yes)
        if enpas != '-':
            emb_idx[-1] = 16

        return emb_idx
    
    def move(self, move, mirror):
        # Get integer representations of move's squares.
        from_square = chess.Move.from_uci(move).from_square
        to_square = chess.Move.from_uci(move).to_square
        # Have to mirror moves if we're playing as black.
        if mirror:
            from_square = chess.square_mirror(from_square)
            to_square = chess.square_mirror(to_square)
        # Initialize tensor that holds indices to move embeddings.
        move_idx = np.array([from_square, to_square])
        # python-chess indexes moves by starting in the last row.
        # We want indices to start in the first row of the board.
        move_idx = (7 - move_idx // 8) * 8 + (move_idx % 8)
        return move_idx

#%%
def sigmoid(x):
    if x<0:
        return np.exp(x) / (1 + np.exp(x))
    else:
        return 1 / (1 + np.exp(-x))

def scorer(score):
    if score.lstrip("-").isdigit():
        return sigmoid(-float(score) * 0.006)

    else:
        match score:
            case 'C':
                return 1.
            case 'I':
                return 0.5
            case 'S':
                return 0.5

#%%
def read_pgn(example, tokenizer):
    pgn = io.StringIO(example['pgn'])
    example['pgn'] = chess.pgn.read_game(pgn).next().eval().relative.score(mate_score=100)
    return example

def read_fen(example, tokenizer):
    fens = example['fens'].split(",")
    max_ply = len(fens) 
    moves = example['moves'].split(",")
    scores = example['scores'].split(",")
    
    rnd = random.randint(min(max_ply-2, 7), max_ply-2)
    #rnd = random.randint(0, max_ply-2)
    fens = fens[rnd]

    pos, turn, _, _ = fens.split(' ')
    if turn == 'b':
        mirror = True
    else:
        mirror = False

    next_idx = rnd+1
    example['fens'] = tokenizer.fen_fast2(fens, mirror)
    example['moves'] = tokenizer.move(moves[next_idx], mirror)
    scores = scores[next_idx].rstrip("\n")
    example['scores'] = scorer(scores)

    return example
