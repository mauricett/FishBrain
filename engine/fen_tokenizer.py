#%%
import torch
import torch.nn as nn


class FEN_tokenizer(nn.Module):
    def __init__(self):
        super(FEN_tokenizer, self).__init__()
        # Define the mapping for pieces and colors
        self.piece_dict = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                           'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
        self.color_dict = {'r': 7, 'n': 7, 'b': 7, 'q': 7, 'k': 7, 'p': 7,
                           'R': 8, 'N': 8, 'B': 8, 'Q': 8, 'K': 8, 'P': 8}
        self.castle_dict = {'K': (0, 10), 'Q': (1, 11),
                            'k': (2, 12), 'q': (3, 13)}

    def __call__(self, fen):
        return self.tokenize(fen)

    def tokenize(self, fen):
        # Initialize the tensors
        params = {'dtype': torch.long, 'pin_memory': True,
                  'requires_grad': False}
        pieces = torch.zeros(size=(8, 8), **params)
        colors = torch.zeros(size=(8, 8), **params)
        castles = torch.full(size=(4,), fill_value=9, **params)
        enpassent = torch.full(size=(1,), fill_value=14, **params)

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

        # En passent is a binary indicator for now (14=no, 15=yes)
        if enpas != '-':
            enpassent[:] = 15

        flat = torch.cat([pieces.flatten(), colors.flatten(), castles, enpassent])
        return flat