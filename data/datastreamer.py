#%%
import chess.pgn

from data.helper import save_metadata, load_metadata, update_metadata
from data.helper import download, decompress


class DataStreamer:
    def __init__(self, resume:      bool = False, 
                       buffer_size: int  = 16 * 10**6):

        self.buffer_size = buffer_size

        self.metadata:   dict
        self.response:   requests.models.Response
        self.pgn:        _io.TextIOWrapper
        self.start_byte: int
        self.worker_id = None

    def init_stream(self):
        self.metadata = load_metadata(self)
        """
        todo:
        handle logic for resume/new stream.
        set self.start_byte after resume
        """

    def start_stream(self, start_byte='0'):
        response = download(self, start_byte, stream=True)
        pgn = decompress(self, response)
        return response, pgn

    def resume_stream(self):
        # Starts a stream at the nearest zstd frame after 'last_byte'.
        print("Resuming stream. Finding entry point for decompression.")
        start_byte = self.metadata['last_byte']
        end_byte = start_byte + 10 * 10**6
        response = download(self, start_byte, end_byte)
        content = response.content
        response.close()
        frame_start = self.find_frame_start(content)
        frame_start_byte = last_byte + frame_start
        response, pgn = self.start_stream(start_byte=frame_start_byte)
        # Discard first game, because the decompressable chunks do not
        # cleanly divide the pgns into legal game data.
        try:
            print("If you receive an error message from python-chess, "
                  "this is harmless. Ignore.")
            print("...")
            _ = chess.pgn.read_game(pgn)
            print("...")
        finally:
            return response, pgn, start_byte

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
        update_metadata(self)
        # Find a pgn which contains evals.
        try:
            while True:
                game = chess.pgn.read_game(self.pgn)

                has_evals = filters.has_evals(game)
                if not has_evals:
                    continue
                
                elo_check = filters.min_elo(game, min_elo)
                if not elo_check:
                    continue

                return game
        except:
            return self.read_game()