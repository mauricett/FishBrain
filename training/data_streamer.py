#%%
import requests
import chess
import chess.pgn
import zstandard as zstd
import io

#%%
class DataStreamer:
    def __init__(self, url=None):


data_streamer = DataStreamer('')