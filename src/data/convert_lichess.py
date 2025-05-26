import io
import time

import zstandard as zstd
import chess
import chess.pgn

from data_models import GameData, GameStruct


MIN_CHARS_PGN = 120
ARCHIVE_PATH  = "raw/2013-01.pgn.zst"


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

    num = {'parsed': 0, 'skipped': 0 }

    with open_zst(ARCHIVE_PATH) as stream:
        for pgn_string in stream_filtered_lines(stream):
            try:
                py_data:  GameData   = extract_game_data(pgn_string)
                c_struct: GameStruct = py_data.to_gamestruct()
                num['parsed'] += 1
            except Exception as e:
                num['skipped'] += 1
                print(f"{num['skipped']}: {e}")
                continue
    
    t2 = time.perf_counter()

    print(f"\ngames skipped: {num['skipped']}")
    print(f"total games stored: {num['parsed']}")
    print(f"total time: {t2 - t1} seconds\n")
