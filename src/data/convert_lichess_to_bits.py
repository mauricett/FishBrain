import zstandard as zstd
import chess
import chess.pgn
import io
import time

path = "raw/2013-01.pgn.zst"
dctx = zstd.ZstdDecompressor()

t1 = time.perf_counter()

n_games = 0
with open(path, "rb") as file:
    reader      = dctx.stream_reader(file)
    file_stream = io.TextIOWrapper(reader)
    line        = file_stream.readline()
    
    while line:
        # get rid of all pgns that are too short or don't have eval
        if (len(line) >= 200) and ("%eval" in line):
            pgn  = io.StringIO(line)
            game = chess.pgn.read_game(pgn)
            n_games += 1

            game = game.next()
            while game:
                game.move
                game.eval()
                if(game.is_end()):
                    game.board().outcome()
                game = game.next()

        line = file_stream.readline()

print(n_games)

t2 = time.perf_counter()
print(t2 - t1)
