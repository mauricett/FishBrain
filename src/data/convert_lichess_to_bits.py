import zstandard as zstd
import chess
import chess.pgn
import io
import time

MIN_CHARS_PGN = 200

class GameSequence:
    scores     = []
    moves      = [] #[ [from, to, is_prmtn], ... ] ???
    outcome    = None
    uci_string = ""

    @classmethod
    def clear_buffers(cls):
        cls.scores     = []
        cls.moves      = []
        cls.outcome    = None
        cls.uci_string = ""

    # something like this?
    #@classmethod
    #def get_struct(cls):
    #    return struct(format_string, ...)

gs = GameSequence()

#%%
path = "raw/2013-01.pgn.zst"
dctx = zstd.ZstdDecompressor()

t1 = time.perf_counter()

n_games = 0
with open(path, "rb") as file:
    reader      = dctx.stream_reader(file)
    file_stream = io.TextIOWrapper(reader)

    for line in file_stream:
        # Only process lines which contain "%eval" and have a min length.
        if (len(line) >= MIN_CHARS_PGN) and ("%eval" in line):
            n_games += 1
            pgn  = io.StringIO(line)
            game = chess.pgn.read_game(pgn)
            
            data = []
            uci_string = ""
            while game:
                """ We need the FENs for each position, but python-chess only
                gives access to that via game.board().fen(), but game.board()
                is slow -> avoid calls to board()!
                Instead, collect moves in uci_string and pass all data to C!
                """
                score    = game.eval()
                move     = game.move.__str__()
                outcome  = game.board().outcome() if game.is_end() else None
                is_prmtn = (move[-1] == 'q')
                
                # need the fen for each position
                # eventually, pass the complete string of moves to a C func
                uci_string += move + ','
                
                data.append([move, score, is_prmtn, outcome])   
                game = game.next()

            #game_sequence = libc.fens(data, uci_string)
    
print(n_games)

t2 = time.perf_counter()
print(t2 - t1)

#%%
data[-2]
uci_string
