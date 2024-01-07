#%%
import os
import requests
import chess
import chess.pgn
import zstandard as zstd
import io
import training.meta_handler as meta


class DataStreamer:
    """
    This class streams data from Lichess to avoid having to download
    entire files at once. Lichess compresses files with zstandard,
    which allows partial decompression. We use this to stream
    and process data on-the-fly.

    The DataStreamer keeps track of the current file and byte position
    by writing to /training/metadata.json. This way we can pause and
    resume training.
    """
    def __init__(self, filepath):
        # zstd frames begin with one of the following two magic numbers.
        self.FRAME_MAGIC_NUMBER = int('0xFD2FB528', 16)
        self.SKIPPABLE_MAGIC_NUMBER = int('0x184D2A50', 16)
        self.file = open(filepath, 'rb')
        self.filepath = filepath
 
    def get_n_bytes(self, start_byte: str, n_range: int) -> bytes:
        start_byte = int(start_byte)
        self.file.seek(start_byte)
        return self.file.read(n_range)
    
    def find_data_frame(self, byte_pos: str) -> str:
        # zstd differentiates between "skippable" and "general" frames.
        # We read the headers' magic numbers to find a general frame.
        magic_number = self.get_n_bytes(byte_pos, n_range=4)
        magic_number = int.from_bytes(magic_number, byteorder='little')
        if magic_number == self.SKIPPABLE_MAGIC_NUMBER:
            byte_pos = self.skip_frame(byte_pos)
            return self.find_data_frame(byte_pos)
        elif magic_number == self.FRAME_MAGIC_NUMBER:
            return byte_pos
        else:
            raise Exception("Could not find frame magic number in zstd file.")
    
    def skip_frame(self, byte_pos: str) -> str:
        header_start = int(byte_pos)
        # Both the magic number and the frame size are encoded in 4 bytes.
        # Skip 4 bytes (magic number) to read frame size encoding:
        frame_size_pos = str(header_start + 4)
        frame_size = self.get_n_bytes(frame_size_pos, n_range=4)
        frame_size = int.from_bytes(frame_size, byteorder='little')
        # Skipping both magic number and frame size encoding (8 bytes)
        # aswell as the skippable content of size frame_size.
        skip_to = header_start + 8 + frame_size
        return str(skip_to)
    
    def check_block_header(self, byte_pos: str):
        # Data frames consist of multiple blocks. To find the total
        # size of a frame, we need to traverse the blocks one by one.
        # We do this by reading the block headers, which are 3 bytes:
        block_header_size = 3
        block_header = self.get_n_bytes(byte_pos, n_range=block_header_size)
        block_header = int.from_bytes(block_header, byteorder='little')
        is_last_block = bin(block_header)[-1]
        block_content_size = block_header >> 3
        next_block_pos = int(byte_pos) + block_header_size + block_content_size
        return int(is_last_block), str(next_block_pos)

    def get_frame_size(self, byte_pos: str) -> int:
        frame_header_size = 6
        block_pos = int(byte_pos) + frame_header_size
        block_pos = str(block_pos)
        # Traverse all blocks till we find last one
        is_last_block = 0
        while not is_last_block:
            is_last_block, block_pos = self.check_block_header(block_pos)
        # Frame size is the last position minus the initial position
        # +4 bytes for checksum after the last block
        frame_size = int(block_pos) - int(byte_pos) + 4
        return str(int(block_pos)+4), frame_size


    def download_frame(self, byte_pos: str) -> list:
        # Headers of data frames are variable size, but seem to always
        # be 6 bytes in the Lichess data. I'm hardcoding this value
        # for now and assume that every data frame ends with a 4 byte
        # checksum. No further info is needed from header with these
        # assumptions. All important info is in the "block headers".
        frame_list = []
        file_size = os.stat(self.filepath).st_size
        byte_pos = '0'
        while int(byte_pos) < file_size:
            byte_pos = self.find_data_frame(byte_pos)
            frame_start = int(byte_pos)
            byte_pos, frame_size = self.get_frame_size(byte_pos)
            frame_list.append((frame_start, frame_size))
        return frame_list

#data_streamer = DataStreamer('data/lichess_db_standard_rated_2013-01.pgn.zst')
data_streamer = DataStreamer('data/lichess_db_standard_rated_2017-01.pgn.zst')
frame_list = data_streamer.download_frame('0')

#%%
dctx = zstd.ZstdDecompressor()
start, size = frame_list[0]
data_streamer.file.seek(start)
x = data_streamer.file.read(size)
y = dctx.stream_reader(x)
z = y.read()

#%%
pgn = z.decode('utf-8')
pgn = io.StringIO(pgn)

#%%
pgn.tell()
pgn.seek(0)
len(z)

g1 = chess.pgn.read_game(pgn)
pgn.getvalue()[pgn.tell():pgn.tell()+600]

#%%
if g1.next().eval():
    print("hey")

#%%
n=0
while n < 256:
    g1 = chess.pgn.read_game(pgn)
    try:
        score = g1.next().eval()
        if score:
            n += 1
    except:
        continue
print(n)
#g1.next().board()

#%%
g1.headers

#%%
pgn = z.decode('utf-8')[0+pgn.tell():200000+pgn.tell()]
pgn = io.StringIO(pgn)

#%%
g1 = chess.pgn.read_game(pgn)
g1.next().eval()
g1.next().board()


#%%
bla = 1
n=0
while bla:
    bla = chess.pgn.read_game(pgn)
    n+=1