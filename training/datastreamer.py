#%%
import requests
import zstandard as zstd
import io
import chess.pgn

class DataStreamer:
    def __init__(self, url, buffer_size=16 * 10**6):
        response = requests.get(url, stream=True)
        data = response.raw
        dctx = zstd.ZstdDecompressor()
        sr = dctx.stream_reader(data, read_size=buffer_size)
        self.pgn = io.TextIOWrapper(sr)

    def read_game(self):
        return chess.pgn.read_game(self.pgn)