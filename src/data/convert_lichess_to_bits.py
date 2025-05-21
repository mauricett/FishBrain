import io
import time

from typing import List
from ctypes import Structure, create_string_buffer
from ctypes import POINTER, c_uint16, c_char, c_int32
from dataclasses import dataclass, field

import zstandard as zstd
import chess
import chess.pgn


MIN_CHARS_PGN = 120
ARCHIVE_PATH  = "raw/2013-01.pgn.zst"
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



def filters_passed(line):
    if len(line) < MIN_CHARS_PGN:
        return False
    if "%eval" not in line:
        return False
    return True


def open_zst(archive_path):
    binary = open(archive_path, "rb")
    dctx = zstd.ZstdDecompressor()
    return io.TextIOWrapper(dctx.stream_reader(binary))


def stream_filtered_lines(io_stream):
    while (line := io_stream.readline()):
        if filters_passed(line):
            yield line


def extract_game_data(pgn_string: str):
    node = chess.pgn.read_game(io.StringIO(pgn_string))
    
    data = GameData()
    # skip first node, holds no data!
    while node := node.next():
        # transform and store as GameData
        data.store_move(node)
        data.store_score(node)
        data.n_moves += 1

        if node.is_end():
            data.store_outcome(node)
            data.has_end = True

    return data


#%%
if __name__ == "__main__":
    print(f"\nparsing archive at {ARCHIVE_PATH}\n")
    print(f"skipped games due to exceptions:")

    t1 = time.perf_counter()

    n_parsed = 0
    n_skipped = 0
    with open_zst(ARCHIVE_PATH) as stream:
        for pgn_string in stream_filtered_lines(stream):
            try:
                py_data:  GameData   = extract_game_data(pgn_string)
                c_struct: GameStruct = py_data.to_gamestruct()
                n_parsed += 1
            except Exception as e:
                n_skipped += 1
                print(f"{n_skipped}: {e}")
                continue
    
    t2 = time.perf_counter()

    print(f"\ngames skipped: {n_skipped}")
    print(f"total games stored: {n_parsed}")
    print(f"total time: {t2 - t1} seconds\n")
