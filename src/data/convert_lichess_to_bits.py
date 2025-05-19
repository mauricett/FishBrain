import zstandard as zstd
import chess
import chess.pgn
import io
import time
import logging
from ctypes import *

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PATH = "raw/2013-01.pgn.zst"
MIN_CHARS_PGN = 200



class GameStruct(Structure):
    _field_ = [
        ("n_moves", c_uint16),
        ("moves", POINTER(c_char)),
        ("scores", POINTER(c_int32))
    ]



def iter_lichess_archive(archive_path, min_chars):
    dctx = zstd.ZstdDecompressor()

    with open(archive_path, "rb") as f:
        reader  = io.BufferedReader(dctx.stream_reader(f))
        text_io = io.TextIOWrapper(reader)

        while (raw_line := text_io.readline()):
            if (len(raw_line) < min_chars) or ("%eval" not in raw_line):
                continue
            try:
                # maybe call chess.pgn.read_game() here?
                game = _parse_game(raw_line)
            except Exception as e:
                logger.warning("Skipping game due to exception:\n%s\n", e)
                continue

            yield game



def _parse_game(pgn_line):
    game_string = io.StringIO(pgn_line)
    root_node = chess.pgn.read_game(game_string)

    moves, scores = [], []
    n_moves = 0

    # skip first node, holds no data!
    node = root_node.next()
    while node:
        if score := node.eval():
            score = score.relative.score(mate_score=100000)
        else:
            # TODO: deal with None scores and mates
            score = 0
        move = node.move.__str__()
        # uci move has length 5 iff promotion, with format a7a8q
        promote = move[-1] if (len(move) == 5) else ''
        """ STILL NEED TO FILTER APPROPRIATE OUTCOMES """
        outcome = node.board().outcome() if node.is_end() else None

        # store data and go to next position in game
        n_moves += 1
        moves.append(move)
        scores.append(score)
        node = node.next()
    
    moves = ",".join(moves)
    return n_moves, moves, scores



#%%
if __name__ == "__main__":
    # benchmark iter_lichess_archive()
    t1 = time.perf_counter()

    n = 0
    for n_moves, moves, scores in iter_lichess_archive(PATH, MIN_CHARS_PGN):
        game = GameStruct(
            n_moves = n_moves,
            moves = create_string_buffer(moves.encode('UTF-8')),
            scores = (n_moves * c_int32)(*scores)
        )
        n += 1

    t2 = time.perf_counter()
    print(t2 - t1)
    print(n)
