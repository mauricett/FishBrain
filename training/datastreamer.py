#%%
import requests
import zstandard as zstd
import io
import chess.pgn

class DataStreamer:
    def __init__(self, url, resume=False, buffer_size=16 * 10**6):
        # resume_stream() will restart the stream only approximately
        # at the location where we stopped, due to technical reasons.
        if not resume:
            self.response, self.pgn = self.new_stream(buffer_size)
        if resume:
            self.response, self.pgn = self.resume_stream(buffer_size)

    def new_stream(self, buffer_size):
        response = requests.get(url, stream=True)
        dctx = zstd.ZstdDecompressor()
        sr = dctx.stream_reader(response.raw, read_size=buffer_size)
        pgn = io.TextIOWrapper(sr)
        return response, pgn

    def read_game(self):
        return chess.pgn.read_game(self.pgn)

    def tell(self):
        return self.response.raw.tell()