#%%
import requests
import io

import zstandard as zstd
import chess.pgn


class DataStreamer:
    def __init__(self, buffer_size: int  = 16 * 10**6):
        self.buffer_size = buffer_size
        # metadata and worker_id are handled by DataLoader
        self.metadata:   dict
        self.worker_id:  str
        self.response:   requests.models.Response
        self.pgn:        _io.TextIOWrapper
        self.start_byte: int
    
    def download(self, start_byte, end_byte='', stream=False):
        url = self.metadata['url']
        headers = {"Range" : "bytes=%s-%s" % (start_byte, end_byte)}
        response = requests.get(url, headers=headers, stream=stream)
        return response

    def decompress(self, response):
        dctx = zstd.ZstdDecompressor()
        sr = dctx.stream_reader(response.raw, read_size=self.buffer_size)
        pgn = io.TextIOWrapper(sr)
        return pgn

    def start_stream(self, start_byte='0'):
        self.start_byte = int(start_byte)
        self.response = self.download(start_byte, stream=True)
        self.pgn = self.decompress(self.response)

    def resume_stream(self):
        # Starts stream at nearest zstd frame after 'consumed_bytes'.
        print("Resuming stream. Finding entry point for decompression.")
        start_byte = self.metadata['consumed_bytes']
        # Frames are ~5 MB large. We download and search the next 10 MB.
        end_byte = start_byte + 10 * 10**6
        response = self.download(start_byte, end_byte)
        content = response.content
        response.close()
        # find_frame_start() returns byte position within the 10 MB file.
        # Add start_byte to get the absolute byte position to resume stream.
        frame_start = start_byte + self.find_frame_start(content)
        self.start_stream(start_byte=frame_start)
        # Discard first game, because frames do not cleanly divide pgns.
        try:
            discard = chess.pgn.read_game(self.pgn)
        finally:
            print("test. appears multiple times? :thonk:")
            pass

    def find_frame_start(self, content):
        # Finds byte position of first decompressable frame.
        FRAME_MAGIC_NUMBER = int('0xFD2FB528', 16)
        for n in range(len(content)):
            number = int.from_bytes(content[n:n+4], byteorder='little')
            if number == FRAME_MAGIC_NUMBER:
                return n
    
    def tell(self):
        # Returns the absolute number of bytes consumed.
        # response.raw.tell() returns the bytes relative
        # to where we restarted the stream. -> Add start_byte.
        return self.response.raw.tell() + self.start_byte