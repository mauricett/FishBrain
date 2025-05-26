from typing import List
from ctypes import Structure, create_string_buffer
from ctypes import POINTER, c_uint16, c_char, c_int32
from dataclasses import dataclass, field

import chess.pgn

# skip games with terminal conditions different from:
LEGAL_OUTCOMES = [
    chess.Termination.CHECKMATE,
    chess.Termination.STALEMATE,
    chess.Termination.INSUFFICIENT_MATERIAL
]


class GameStruct(Structure):
    _fields_ = [
        ("n_moves", c_uint16),
        ("moves",   POINTER(c_char)),
        ("scores",  POINTER(c_char))
    ]


@dataclass
class GameData:
    moves:   List[str] = field(default_factory=list)
    scores:  List[str] = field(default_factory=list)
    has_end: bool = False
    outcome: int  = 0
    n_moves: int  = 0


    def store_score(self, node: chess.pgn.ChildNode):
        if score := node.eval():
            score = score.pov(color=chess.WHITE)
        else:
            # missing eval only allowed if is_end
            if node.is_end():
                score = 'X'
            else:
                raise Exception("missing sf eval")
        self.scores.append(str(score))


    def store_move(self, node: chess.pgn.ChildNode):
        if node.move:
            self.moves.append(str(node.move))
        else:
            raise Exception("move could not be read", node.move)


    def store_outcome(self, node: chess.pgn.ChildNode):
        outcome = node.end().board().outcome()
        if outcome:
            if outcome.termination in LEGAL_OUTCOMES:
                self.outcome = outcome.termination.value
            else:
                raise Exception("skipping termination: ", outcome.termination.value)
        else:
            if node.end().eval():
                self.outcome = 0
            else:
                raise Exception("missing both termination and sf eval")


    def store_outcome(self, node: chess.pgn.ChildNode):
        if outcome := node.end().board().outcome():
            if outcome.termination in LEGAL_OUTCOMES:
                self.outcome = outcome.termination.value
            else:
                raise Exception("skipping termination: ", outcome.termination.value)
        else:
            if node.end().eval():
                self.outcome = 0
            else:
                raise Exception("missing both termination and sf eval")


    def to_gamestruct(self):
        if not self.has_end:
            raise Exception("missing end node in extract_game_data()")
        
        if not (self.n_moves == len(self.moves) == len(self.scores)):
            raise Exception("mismatch in count and array sizes")

        moves_str = ",".join(self.moves)
        moves_str = moves_str.encode('UTF-8')
        
        scores_str = ",".join(self.scores)
        scores_str = scores_str.encode('UTF-8')

        return GameStruct(
            n_moves = self.n_moves,
            moves   = create_string_buffer(moves_str),
            scores  = create_string_buffer(scores_str)
        )
