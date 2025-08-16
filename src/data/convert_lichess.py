#%%
import time
from ctypes import *
from data_models import ZstReader


libc = CDLL("./lib.o")

MIN_LENGTH_PGN = 120
ZST_ARCHIVE_PATH  = "raw/2013-01.pgn.zst"

zst_reader = ZstReader(
    min_length_pgn   = MIN_LENGTH_PGN,
    zst_archive_path = ZST_ARCHIVE_PATH
)

if __name__ == "__main__":
    print(f"\nparsing archive at {ZST_ARCHIVE_PATH}\n")
    print(f"skipped games due to exceptions:")

    t1 = time.perf_counter()

    for game in zst_reader.iter_pgns():
        try:
            data = zst_reader.parse_pgn(game)
            #buffer.add_game(data)
            #num['parsed'] += 1
        except Exception as e:
            #num['skipped'] += 1
            #print(f"{num['skipped']}: {e}")
            continue

    t2 = time.perf_counter()

    #print(f"\ngames skipped: {num['skipped']}")
    #print(f"total games stored: {num['parsed']}")
    print(f"total time: {t2 - t1} seconds\n")
