import zstandard as zstd
import chess
import chess.pgn
import io
import time
import logging


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PATH = "raw/2013-01.pgn.zst"
MIN_CHARS_PGN = 200



class GameSequence:
    scores     = []
    moves      = [] #[ [from, to, promote], ... ] ???
    outcome    = None
    uci_moves = ""

    @classmethod
    def clear_buffers(cls):
        cls.scores     = []
        cls.moves      = []
        cls.outcome    = None
        cls.uci_moves = ""

    # something like this?
    #@classmethod
    #def get_struct(cls):
    #    return struct(format_string, ...)



def convert_lichess_archive(archive_path, min_chars):
    dctx = zstd.ZstdDecompressor()
    n_games = 0

    with open(archive_path, "rb") as file:
        reader = dctx.stream_reader(file)
        text_io = io.TextIOWrapper(reader)

        for line in text_io:
            is_not_relevant = (len(line) < min_chars) or ("%eval" not in line)

            if is_not_relevant:
                continue
            try:
                data, uci_moves = _parse_game(line)
                n_games += 1
            except Exception as e:
                logging.warning("Exception while parsing game:\n%s\n", e)

    return n_games



def iter_lichess_archive(archive_path, min_chars):
    dctx = zstd.ZstdDecompressor()

    with open(archive_path, "rb") as f:
        reader  = io.BufferedReader(dctx.stream_reader(f))
        text_io = io.TextIOWrapper(reader)

        while (raw_line := text_io.readline()):
            if (len(raw_line) < min_chars) or ("%eval" not in raw_line):
                continue
            try:
                game = _parse_game(raw_line)
            except Exception as e:
                logger.warning("Skipping game due to exception:\n%s\n", e)
                continue

            yield game



def _parse_game(pgn_line):
    game_string = io.StringIO(pgn_line)
    root_node   = chess.pgn.read_game(game_string)

    data, uci_moves = [], []
    
    node = root_node
    while node:
        if score := node.eval():
            score = score.relative.score(mate_score=100000)
        move = node.move.__str__()
        # uci move has length 5 iff promotion, with format a7a8q
        promote = move[-1] if (len(move) == 5) else ''

        """ STILL NEED TO FILTER APPROPRIATE OUTCOMES """
        outcome = node.board().outcome() if node.is_end() else None

        # store data and go to next position in game
        uci_moves.append(move)
        data.append([score, move, promote, outcome])
        node = node.next()
    
    #game_sequence = libc.fens(data, ",".join(uci_moves))
    return data, ",".join(uci_moves)
 

#%%
if __name__ == "__main__":
    # benchmark convert_lichess_archive()
    t1 = time.perf_counter()
    n_games = convert_lichess_archive(PATH, MIN_CHARS_PGN)
    t2 = time.perf_counter()
    print(t2 - t1)
    print(n_games)

    # benchmark iter_lichess_archive()
    t1 = time.perf_counter()
    n = 0
    for x, y in iter_lichess_archive(PATH, MIN_CHARS_PGN):
        n += 1
    t2 = time.perf_counter()
    print(t2 - t1)
    print(n)

#%%
x
