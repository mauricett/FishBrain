#%%
import requests
import json

import chess.pgn
from torch.utils.data import IterableDataset

from data.datastreamer import DataStreamer
from data.helper import save_metadata, load_metadata, update_metadata
import data.filters as filters


class LichessData(IterableDataset):
    def __init__(self, min_elo, resume):
        super(LichessData, self).__init__()
        self.min_elo = min_elo
        self.resume = resume
        self.data_streamer: data.datastreamer.DataStreamer

    def get_data_url(self, worker_id):
        link_list = requests.get("https://database.lichess.org/standard/list.txt")
        lines = link_list.text.splitlines()
        return lines[worker_id]

    def config(self, worker_id):
        self.data_streamer = DataStreamer()
        self.data_streamer.worker_id = worker_id

        if not self.resume:
            data_url = self.get_data_url(worker_id)
            metadata = {"url" : data_url, "consumed_bytes" : 0}
            self.data_streamer.metadata = metadata
            save_metadata(self.data_streamer)
            self.data_streamer.start_stream()

        if self.resume:
            load_metadata(self.data_streamer)
            self.data_streamer.resume_stream()

    def read_game(self):
        # Find a pgn which contains evals.
        try:
            while True:
                game = chess.pgn.read_game(self.data_streamer.pgn)

                has_evals = filters.has_evals(game)
                if not has_evals:
                    continue
                
                elo_check = filters.min_elos(game, self.min_elo)
                if not elo_check:
                    continue

                return game
        except:
            return self.read_game()

    def __iter__(self):
        while True:
            game = self.read_game()
            update_metadata(self.data_streamer)
            yield int(game.headers['WhiteElo'])