#%%
import chess.pgn

import data.filters as filters
from data.helper import download, decompress


class DataStreamer:
    def __init__(self, resume: bool = False, buffer_size: int  = 16 * 10**6):
        self.buffer_size = buffer_size
        # metadata and worker_id are handled by DataLoader
        self.metadata:   dict
        self.worker_id:  str
        self.response:   requests.models.Response
        self.pgn:        _io.TextIOWrapper
        self.start_byte: int

    def start_stream(self, start_byte='0'):
        self.start_byte = int(start_byte)
        self.response = download(self, start_byte, stream=True)
        self.pgn = decompress(self, self.response)

    def resume_stream(self):
        # Starts stream at nearest zstd frame after 'consumed_bytes'.
        print("Resuming stream. Finding entry point for decompression.")
        start_byte = self.metadata['consumed_bytes']
        # Frames are ~5 MB large. We download and search the next 10 MB.
        end_byte = start_byte + 10 * 10**6
        response = download(self, start_byte, end_byte)
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
    
    def read_game(self, min_elo=0):
        # Find a pgn which contains evals.
        try:
            while True:
                game = chess.pgn.read_game(self.pgn)

                has_evals = filters.has_evals(game)
                if not has_evals:
                    continue
                
                elo_check = filters.min_elos(game, min_elo)
                if not elo_check:
                    continue

                return game
        except:
            return self.read_game(min_elo)