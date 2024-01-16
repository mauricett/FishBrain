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

    def random_ply(self, exclude_first=3, exclude_last=1):
        # pick random ply, excluding some number of early and late plys
        n_moves = self.count_moves()
        n_moves = random.randint(exclude_first, n_moves - exclude_last)
        for n in range(n_moves):
            self.game = self.game.next()
        return self.game

    def fen(self):
        return self.game.board().fen()

    def legal_moves(self):
        moves = []
        for move in self.game.board().legal_moves:
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