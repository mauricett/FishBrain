import zstandard as zstd
import chess
import chess.pgn
import io
import time

MIN_CHARS_PGN = 200

#%%
path = "raw/2013-01.pgn.zst"
dctx = zstd.ZstdDecompressor()

t1 = time.perf_counter()

n_games = 0
with open(path, "rb") as file:
    reader      = dctx.stream_reader(file)
    file_stream = io.TextIOWrapper(reader)

    for line in file_stream:
        # check min length for PGN, then check if PGN contains SF eval
        if (len(line) >= MIN_CHARS_PGN) and ("%eval" in line):
            n_games += 1
            pgn  = io.StringIO(line)
            game = chess.pgn.read_game(pgn)
            
            # game_sequence, board = process_moves(game)

            data = []
            uci_string = ""
            while game:
                """ We try to avoid calling game.board(), instead
                    pass complete sequence of moves to C function.
                    We only call game.board() once, if is_end().
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

            # board_sequence = libc.fens(data, uci_string)
    
print(n_games)

t2 = time.perf_counter()
print(t2 - t1)

#%%
data[-2]
uci_string
