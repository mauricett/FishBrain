#%%
import time
from ctypes import *
from zst_reader import ZstReader
from buffer_processor import BufferProcessor


MIN_LENGTH_PGN   = 120
N_CHUNKSIZE      = 10**2
#ZST_ARCHIVE_PATH = "../data/raw/2025-07.pgn.zst"
ZST_ARCHIVE_PATH = "raw/2013-01.pgn.zst"


libc = CDLL("./lib.o")

buffer = BufferProcessor(
    games_per_file = N_CHUNKSIZE,
    move_separator = ",",
    game_separator = ";",
    c_processor    = libc.process
)

zst_reader = ZstReader(
    min_length_pgn   = MIN_LENGTH_PGN,
    zst_archive_path = ZST_ARCHIVE_PATH,
    buffer           = buffer
)


if __name__ == "__main__":
    print(f"\nparsing archive at {ZST_ARCHIVE_PATH}\n")
    print(f"skipped games due to exceptions:")
    t1 = time.perf_counter()


    for game in zst_reader.iter_pgns():
        try:
            data = zst_reader.parse_pgn(game)
            buffer.add_game(data)

            if buffer.is_full:
                print("buffer full, writing chunk")
                buffer.process_and_clear()

        except Exception as e:
            print(f"Skipping pgn due to exception: {e}")
            continue

    if not buffer.is_empty:
        buffer.process_and_clear()


    t2 = time.perf_counter()
    #print(f"\ngames skipped: {num['skipped']}")
    #print(f"total games stored: {num['parsed']}")
    print(f"total time: {t2 - t1} seconds\n")
