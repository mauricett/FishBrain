#%%
import chess
import torch
import torch.nn as nn


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
    
    def move(self, move, mirror):
        #move = game_node.next().move
        # Initialize tensor that holds indices to move embeddings.
        move_idx = torch.zeros(size=(2,))
        # Bring move into UCI format for python-chess functions.
        move = move.uci()
        # Get integer representations of move's squares.
        from_square = chess.Move.from_uci(move).from_square
        to_square = chess.Move.from_uci(move).to_square
        # Have to mirror moves if we're playing as black.
        if mirror:
            from_square = chess.square_mirror(from_square)
            to_square = chess.square_mirror(to_square)
        move_idx[:] = torch.tensor([from_square, to_square])
        # python-chess indexes moves by starting in the last row.
        # We want indices to start in the first row of the board.
        move_idx = (7 - move_idx // 8) * 8 + (move_idx % 8)
        return move_idx

    def fen(self, game_node, mirror):
        if mirror:
            fen = game_node.board().mirror().fen()
        else:
            fen = game_node.board().fen()

        # Initialize tensors that hold indices for embedding tokens.
        pieces = torch.zeros(size=(8, 8))
        colors = torch.full(size=(8, 8), fill_value=9)
        castles = torch.full(size=(4,), fill_value=14)
        enpassent = torch.full(size=(1,), fill_value=15)

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

        # En passent is a binary indicator for now (15=no, 16=yes)
        if enpas != '-':
            enpassent[:] = 16

        flat = torch.cat([pieces.flatten(), colors.flatten(), castles, enpassent])
        return flat