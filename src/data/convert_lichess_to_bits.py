import zstandard as zstd
import chess
import chess.pgn
import io
import time
import logging
from typing import List
from dataclasses import dataclass, field
from ctypes import *


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ARCHIVE_PATH = "raw/2013-01.pgn.zst"
MIN_CHARS_PGN = 120



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
    outcome: str = ""
    n_moves: int = 0


    def store_score(self, node: chess.pgn.ChildNode):
        """ STILL NEED TO DEAL WITH NONE AND MATES """
        if score := node.eval():
            score = score.relative.score(mate_score=100000)
        else:
            score = 0
        self.scores.append(score)


    def store_move(self, node: chess.pgn.ChildNode):
        """ STILL NEED TO DEAL WITH NONE """
        move = node.move.__str__()
        self.moves.append(move)
        

    def store_outcome(self, node: chess.pgn.ChildNode):
        """ STILL NEED TO FILTER APPROPRIATE OUTCOMES """
        #outcome = node.board().outcome() if node.is_end() else None
        #self.outcome = outcome
        if node.is_end():
            self.outcome = node.board().outcome()
            print(self.outcome)


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
    dctx = zstd.ZstdDecompressor()
    binary = open(archive_path, "rb")
    return io.TextIOWrapper(dctx.stream_reader(binary))


def stream_filtered_lines(io_stream, min_chars_pgn):
    while (line := io_stream.readline()):
        if filters_passed(line, min_chars_pgn):
            yield line


def extract_game_data(pgn_string: str):
    node = chess.pgn.read_game(io.StringIO(pgn_string))
    
    game_data = GameData()
    # skip first node, holds no data!
    while node := node.next():
        # transform and store as GameData
        game_data.store_move(node)
        game_data.store_score(node)
        game_data.store_outcome(node)
        game_data.n_moves += 1

    return game_data


#%%
if __name__ == "__main__":

    t1 = time.perf_counter()

    n = 0
    with open_zst(ARCHIVE_PATH) as stream:
        for pgn_string in stream_filtered_lines(stream, MIN_CHARS_PGN):
            game_data   = extract_game_data(pgn_string)
            game_struct = game_data.to_gamestruct()
            n += 1

    t2 = time.perf_counter()

    print(t2 - t1)
    print(n)

#%%
game_data
