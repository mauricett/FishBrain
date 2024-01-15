#%%
import torch


class FEN_tokenizer(nn.Module):
    def __init__(self):
        super(FEN_tokenizer, self).__init__()
        # Define the mapping for pieces and colors
        self.piece_dict = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                           'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6}
        self.color_dict = {'r': 2, 'n': 2, 'b': 2, 'q': 2, 'k': 2, 'p': 2,
                           'R': 1, 'N': 1, 'B': 1, 'Q': 1, 'K': 1, 'P': 1}

    def __call__(self, fen):
        return self.piece_embed(fen)

    def piece_embed(self, fen):
        # Initialize the tensors
        pieces = torch.zeros(8, 8, dtype=torch.long)
        colors = torch.zeros(8, 8, dtype=torch.long)

        # Split the FEN string into rows       
        position, turn, castling, _, _, _ = fen.split(' ')
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

        return pieces, colors