#%%
import io
from typing import List
from dataclasses import dataclass, field

from zstandard import ZstdDecompressor
import chess.pgn



@dataclass
class GameData:
    moves:   List[str] = field(default_factory=list)
    evals:  List[str] = field(default_factory=list)
    has_end: bool = False
    outcome: int  = 0
    n_moves: int  = 0



class ZstReader:
    def __init__(self, min_length_pgn, zst_archive_path, buffer):
        self.min_length_pgn = min_length_pgn
        self.zst_archive_path = zst_archive_path
        self.buffer = buffer
        self.n_parsed_games = 0
        self.n_total_games = 0

        self.legal_outcomes = {
            chess.Termination.CHECKMATE: 32000,
            chess.Termination.STALEMATE: 31000,
            chess.Termination.INSUFFICIENT_MATERIAL: 31001
        }

    def pass_filters(self, line):
        if len(line) < self.min_length_pgn:
            return False
        if "%eval" not in line:
            return False
        return True


    def iter_pgns(self):
        with open(self.zst_archive_path, "rb") as file:
            dctx = ZstdDecompressor()
            reader = dctx.stream_reader(file)
            stream = io.TextIOWrapper(reader)
            while (line := stream.readline()):
                if self.pass_filters(line):
                    yield line
    

    def parse_pgn(self, pgn_string):
        self.n_total_games += 1
        
        node = chess.pgn.read_game(io.StringIO(pgn_string))
        data = GameData()
        # this skips first node, which only contains the starting position
        while node := node.next():
            # transform and store as GameData
            self._store_move(node, data)

            if not node.is_end():
                self._store_score(node, data)
            else:
                self._store_outcome(node, data)
                data.has_end = True

        if not data.has_end:
            raise Exception("missing end node in parge_pgn()")
        
        if not (data.n_moves == len(data.moves) == len(data.evals)):
            raise Exception("mismatch in count and array sizes")

        self.n_parsed_games += 1
        return data
    

    def _store_score(self, node: chess.pgn.ChildNode, data: GameData):
        if score := node.eval():
            score = score.pov(color=chess.WHITE)
            score = score.score(mate_score=32000)
            data.evals.append(str(score))
        else:
            raise Exception("missing sf eval")
    

    def _store_outcome(self, node: chess.pgn.ChildNode, data: GameData):
        outcome = node.board().outcome()
        if outcome:
            if outcome.termination in self.legal_outcomes.keys():
                data.evals.append(str(self._encode(outcome)))
            else:
                raise Exception(f"skipping termination: {outcome}")
        elif node.eval():
            self._store_score(node, data)
        else:
            raise Exception(f"missing both termination and sf eval")


    def _store_move(self, node: chess.pgn.ChildNode, data: GameData):
        if node.move:
            data.moves.append(str(node.move))
            data.n_moves += 1
        else:
            raise Exception("move could not be read", node.move)


    def _encode(self, outcome):
        if outcome.termination == chess.Termination.CHECKMATE:
            if not outcome.winner:
                return -self.legal_outcomes[outcome.termination]
        return self.legal_outcomes[outcome.termination]
