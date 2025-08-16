import io
from typing import List
from dataclasses import dataclass, field

from zstandard import ZstdDecompressor
import chess.pgn


# skip games with terminal conditions different from:
LEGAL_OUTCOMES = [
    chess.Termination.CHECKMATE,
    chess.Termination.STALEMATE,
    chess.Termination.INSUFFICIENT_MATERIAL
]


@dataclass
class GameData:
    moves:   List[str] = field(default_factory=list)
    scores:  List[str] = field(default_factory=list)
    has_end: bool = False
    outcome: int  = 0
    n_moves: int  = 0


class ZstReader:
    def __init__(self, min_length_pgn, zst_archive_path):
        self.min_length_pgn = min_length_pgn
        self.zst_archive_path = zst_archive_path
 

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
        node = chess.pgn.read_game(io.StringIO(pgn_string))
        
        data = GameData()
        # skip first node, contains only the starting position
        while node := node.next():
            # transform and store as GameData
            self._store_move(node, data)
            self._store_score(node, data)

            if node.is_end():
                self._store_outcome(node, data)
                data.has_end = True

        if not data.has_end:
            raise Exception("missing end node in extract_game_data()")
        
        if not (data.n_moves == len(data.moves) == len(data.scores)):
            raise Exception("mismatch in count and array sizes")

        return data
    

    def _store_score(self, node: chess.pgn.ChildNode, data: GameData):
        if score := node.eval():
            score = score.pov(color=chess.WHITE)
        else:
            # missing eval only allowed if is_end
            if node.is_end():
                score = 'X'
            else:
                raise Exception("missing sf eval")
        data.scores.append(str(score))


    def _store_move(self, node: chess.pgn.ChildNode, data: GameData):
        if node.move:
            data.moves.append(str(node.move))
            data.n_moves += 1
        else:
            raise Exception("move could not be read", node.move)


    def _store_outcome(self, node: chess.pgn.ChildNode, data: GameData):
        outcome = node.end().board().outcome()
        if outcome:
            if outcome.termination in LEGAL_OUTCOMES:
                data.outcome = outcome.termination.value
            else:
                raise Exception("skipping termination: ", outcome.termination.value)
        else:
            if node.end().eval():
                data.outcome = 0
            else:
                raise Exception("missing both termination and sf eval")



