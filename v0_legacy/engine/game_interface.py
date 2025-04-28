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
        # Reset game to initial state
        self.game = self.game.game()
        # pick random ply, excluding some number of early and late plys
        n_moves = self.count_moves()
        n_moves = random.randint(exclude_first, n_moves - exclude_last)
        for n in range(n_moves):
            self.game = self.game.next()
        return self.game

    def fen(self, mirror) -> str:
        match mirror:
            case True:
                return self.game.board().mirror().fen()
            case False:
                return self.game.board().fen()
        
    def next_move(self):
        return self.game.next().move
    
    def next_eval(self):
        return self.game.next().eval()
    
    def turn(self):
        return self.game.board().turn
        
    def legal_moves(self):
        return list(self.game.board().legal_moves)

    def show(self, move_n):
        game = self.game.game()
        for n in range(move_n):
            game = game.next()
        return game