#%%
import time
from ctypes import *
from zst_reader import ZstReader
from buffer_processor import BufferProcessor


MIN_LENGTH_PGN = 120
ZST_ARCHIVE_PATH  = "raw/2013-01.pgn.zst"


libc = CDLL("./lib.o")

zst_reader = ZstReader(
    min_length_pgn   = MIN_LENGTH_PGN,
    zst_archive_path = ZST_ARCHIVE_PATH
)

buffer = BufferProcessor()

if __name__ == "__main__":
    print(f"\nparsing archive at {ZST_ARCHIVE_PATH}\n")
    print(f"skipped games due to exceptions:")

    t1 = time.perf_counter()


    for game in zst_reader.iter_pgns():
        try:
            data = zst_reader.parse_pgn(game) # increments 'total_read' at beginning of parse_pgn(), increment 'parsed' at end.

            buffer.add_game(data) # stores 'buffered', 'total_saved'
            if buffer.is_full:
                buffer.process()

        except Exception as e:
            continue

    if not buffer.is_empty:
        buffer.process_final()


    t2 = time.perf_counter()

    #print(f"\ngames skipped: {num['skipped']}")
    #print(f"total games stored: {num['parsed']}")
    print(f"total time: {t2 - t1} seconds\n")
