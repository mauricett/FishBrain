import zstandard as zstd
import chess
import chess.pgn
import io
import time
import logging
from typing import List
from dataclasses import dataclass, field
from ctypes import Structure, create_string_buffer
from ctypes import POINTER, c_uint16, c_char, c_int32

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ARCHIVE_PATH = "raw/2013-01.pgn.zst"
MIN_CHARS_PGN = 120

# values from chess.Termination class:
# checkmate  == 1
# stalemate  == 2
# insuff mat == 3
# we skip games with outcomes > 3.
LEGAL_OUTCOMES = [0, 1, 2, 3] 



class GameStruct(Structure):
    _fields_ = [
        ("n_moves", c_uint16),
        ("moves",   POINTER(c_char)),
        ("scores",  POINTER(c_int32))
    ]


@dataclass
class GameData:
    moves:   List[str] = field(default_factory=list)
    scores:  List[int] = field(default_factory=list)
    has_end: bool = False
    outcome: int  = 0
    n_moves: int  = 0

    def store_score(self, node: chess.pgn.ChildNode):
        """ STILL NEED TO DEAL WITH NONE AND MATES """
        if score := node.eval():
            score = score.relative.score(mate_score=100000)
        else:
            score = 0
        self.scores.append(score)

    def store_move(self, node: chess.pgn.ChildNode):
        if node.move:
            self.moves.append(node.move.__str__())
        else:
            raise Exception("move could not be read")
        
    def store_outcome(self, node: chess.pgn.ChildNode):
        """ read the terminal condition from final game node.
        raise exception if the node misses both SF eval and outcome.
        """
        if outcome := node.end().board().outcome():
            self.outcome = outcome.termination.value
        else:
            if not node.end().eval():
                raise Exception("missing both outcome condition and sf eval")
            self.outcome = 0

    def to_gamestruct(self):
        moves_str = ",".join(self.moves)
        moves_str = moves_str.encode('UTF-8')
        return GameStruct(
            n_moves = self.n_moves,
            moves   = create_string_buffer(moves_str),
            scores  = (self.n_moves * c_int32)(*self.scores)
        )


def filters_passed(line, min_chars_pgn):
    if len(line) < min_chars_pgn:
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
        if filters_passed(line, MIN_CHARS_PGN):
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

            if data.outcome not in LEGAL_OUTCOMES:
                raise Exception("illegal termination condition")
    
    if not data.has_end:
        raise Exception("missing end node in extract_game_data()")
    
    if not (data.n_moves == len(data.moves) == len(data.scores)):
        raise Exception("mismatch in count and array sizes")

    return data


#%%
if __name__ == "__main__":

    t1 = time.perf_counter()

    n = 0
    with open_zst(ARCHIVE_PATH) as stream:
        for pgn_string in stream_filtered_lines(stream):
            try:
                py_data:  GameData   = extract_game_data(pgn_string)
                c_struct: GameStruct = py_data.to_gamestruct()
                n += 1
            except Exception as e:
                logger.warning("\nskip current line, because:\n%s\n", e)
                continue
    
    t2 = time.perf_counter()

    print(t2 - t1)
    print(n)
#%%
