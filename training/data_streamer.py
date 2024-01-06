#%%
import requests
import chess
import chess.pgn
import zstandard as zstd
import io
import training.meta_handler as meta


class DataStreamer:
    """
    This class can stream data from Lichess without having to
    download entire files. Lichess compresses files with zstandard,
    which allows partial decompression. We use this to stream
    and process data on-the-fly.

    The DataStreamer keeps track of the current file and byte position
    by writing to /training/metadata.json. This way we can pause and
    resume training.
    """
    def __init__(self):
        # zstd frames begin with one of the following two magic numbers.
        self.FRAME_MAGIC_NUMBER = int('0xFD2FB528', 16)
        self.SKIPPABLE_MAGIC_NUMBER = int('0x184D2A50', 16)
        
        self.meta_handler = meta.Handler()
        self.metadata = self.meta_handler.load()
        self.url = self.metadata['url']
        self.current_pos = self.metadata['current_pos']
    
    def find_next_frame(self, byte_pos: str) -> str:
        magic_number = self.get_n_bytes(byte_pos, n_range=4)
        magic_number = int.from_bytes(magic_number, byteorder='little')
        if magic_number == self.SKIPPABLE_MAGIC_NUMBER:
            byte_pos = self.skip_frame(byte_pos)
            return self.find_next_frame(byte_pos)
        elif magic_number == self.FRAME_MAGIC_NUMBER:
            return byte_pos
        else:
            raise Exception("Could not find frame magic number in zstd file.")

    def get_n_bytes(self, start_byte: str, n_range: int) -> bytes:
        start_byte = int(start_byte)
        # Range in requests.get() is inclusive, we need to subtract 1.
        end_byte = start_byte + n_range-1
        request_head = {"Range":"bytes=%s-%s" % (start_byte, end_byte)}
        response = requests.get(self.url, headers=request_head)
        content = response.content
        return content
    
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
    
    def download_frame(self, byte_pos: str) -> bytes:
        # Headers of data frames are variable size, but seem to always
        # be 6 bytes in the Lichess data. I'm hardcoding this value
        # for now and assume that every data frame ends with a 4 byte
        # checksum. No further data is needed from header with these
        # assumptions. All important info is in the "block headers".
        block_pos = int(byte_pos) + 6

    def get_data_frame_size(self):
        # Data frames consist of multiple blocks. To find the total
        # size of the frame, we need to traverse the blocks one by one
        # and sum up their sizes, until we reach the end of the frame.
        # Once we know the size, we can download and decompress the frame.
        
        pass


data_streamer = DataStreamer()

#%%
#url = "https://database.lichess.org/standard/lichess_db_standard_rated_2023-11.pgn.zst"
#response = requests.head(url)
#response.headers['Content-Length']