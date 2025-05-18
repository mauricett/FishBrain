import zstandard as zstd
import chess
import chess.pgn
import io
import time
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
print("__name__ = ", __name__)


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

gs = GameSequence()


# should i rewrite this with yield?
# yield results for single games, and write result to disk in different fn?
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
                logging.warning("Excepction while parsing game:\n%s\n", e)

    return n_games



def _parse_game(pgn_line):
    game_string = io.StringIO(pgn_line)
    root_node   = chess.pgn.read_game(game_string)

    data, uci_moves = [], []
    
    node = root_node
    while node:
        score   = game.eval()
        move    = game.move.__str__()
        # uci move has length 5 iff promotion, with format a7a8q
        promote = move[-1] if (len(move) == 5) else ''
        outcome = game.board().outcome() if game.is_end() else None

        # store results
        uci_moves.append(move)
        data.append([score, move, promote, outcome])

        # go to next position in the game sequence
        node = node.next()
    
    #game_sequence = libc.fens(data, ",".join(uci_moves))
    return data, ",".join(uci_moves)
 


if __name__ == "__main__":
    
    t1 = time.perf_counter()
    n_games = convert_lichess_archive(PATH, MIN_CHARS_PGN)
    t2 = time.perf_counter()
    
    print(t2 - t1)
    print(n_games)
