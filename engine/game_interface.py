#%%
import random
import chess
import chess.pgn
import data.filters as filters


class GameInterface():
    def __init__(self):
        self.game = None
        self.n_moves = None

    def count_moves(self):
        n_moves = self.game.end().ply()
        return n_moves

    def forward_random_(self):
        # pick random board position except the final position
        n_moves = random.randint(0, self.n_moves - 1)
        for n in range(n_moves):
            self.next()
        return self.game

    def fen(self):
        return self.game.board().fen()

    def legal_moves(self):
        moves = []
        for move in self.board().legal_moves:
            moves.append(move)
        return moves

    def read_game(self, pgn, min_elo):
        # Find a pgn which contains evals.
        while True:
            try:
                game = chess.pgn.read_game(pgn)

                has_evals = filters.has_evals(game)
                if not has_evals:
                    continue
                elo_check = filters.min_elos(game, min_elo)
                if not elo_check:
                    continue
            except:
                continue
            return game