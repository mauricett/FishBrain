#%%
import torch
import torch.nn as nn


class FEN_tokenizer(nn.Module):
    def __init__(self):
        super(FEN_tokenizer, self).__init__()
        # Define the mapping for pieces and colors
        self.piece_dict = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                           'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
        self.color_dict = {'r': 2, 'n': 2, 'b': 2, 'q': 2, 'k': 2, 'p': 2,
                           'R': 1, 'N': 1, 'B': 1, 'Q': 1, 'K': 1, 'P': 1}
        self.castle_dict = {'K': (0, 1), 'Q': (1, 2),
                            'k': (2, 3), 'q': (3, 4)}

    def __call__(self, fen):
        return self.tokenize(fen)

    def tokenize(self, fen):
        # Initialize the tensors
        pieces = torch.zeros(8, 8, dtype=torch.int, pin_memory=True)
        colors = torch.zeros(8, 8, dtype=torch.int, pin_memory=True)
        castles = torch.zeros(4, dtype=torch.int, pin_memory=True)
        enpassent = torch.zeros(1, dtype=torch.int, pin_memory=True)

        # Split FEN string into 
        position, turn, castling, enpas, _, _ = fen.split(' ')
        rows = position.split('/')

        # Iterate over the rows and columns and fill the tensors
        for i, row in enumerate(rows):
            col = 0
            for char in row:
                if char.isdigit():
                    # If the character is a digit, skip that many columns
                    col += int(char)
                else:
                    # Otherwise, fill the tensors with the piece and color values
                    pieces[i, col] = self.piece_dict[char]
                    colors[i, col] = self.color_dict[char]
                    col += 1
               
        # Each castling right gets a unique embedding token
        if castling != '-':
            for char in castling:
                i, embed_token = self.castle_dict[char]
                castles[i] = embed_token

        # En passent is a binary indicator for now
        if enpas != '-':
            enpassent[:] = 1

        return pieces, colors, castles, enpassent