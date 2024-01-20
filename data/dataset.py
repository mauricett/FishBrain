#%%
import requests
import json

import chess.pgn
from torch.utils.data import IterableDataset, get_worker_info

import data.filters as filters
from data.datastreamer import DataStreamer
from data.helper import save_metadata, load_metadata, update_metadata
from engine.game_interface import GameInterface
from engine.fen_tokenizer import FEN_tokenizer


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.config(worker_id)

class LichessData(IterableDataset):
    def __init__(self, min_elo, resume):
        super(LichessData, self).__init__()
        self.min_elo = min_elo
        self.resume = resume
        self.chess = GameInterface()
        self.tokenizer = FEN_tokenizer()
        self.data_streamer: data.datastreamer.DataStreamer

    def get_data_url(self, worker_id):
        with open("data/metadata/list.txt", "r") as file:
            link_list = file.read()
            lines = link_list.splitlines()
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
        while True:
            # Wrap read_game() in try/except to skip dirty data.
            try:
                game = chess.pgn.read_game(self.data_streamer.pgn)

                # Check if game has SF evals and a minimum Elo or skip.
                has_evals = filters.has_evals(game)
                if not has_evals:
                    continue
                elo_check = filters.min_elos(game, self.min_elo)
                if not elo_check:
                    continue

            except:
                continue
            return game

    def __iter__(self):
        while True:
            # Load a game into our GameInterface self.chess.
            self.chess.game = self.read_game()
            # Update number of bytes consumed by streamer.
            update_metadata(self.data_streamer)

            # Randomly pick a chess position from the game.
            self.chess.game = self.chess.random_ply()
            # Get next move and corresponding score.
            move = self.chess.game.next().move
            score = self.chess.game.next().eval()

            # If black's turn, use a mirrored and color-flipped board.
            # This makes it so every position looks like it was white's
            # perspective and turn.
            turn = self.chess.game.board().turn
            match turn:
                case chess.WHITE:
                    mirror = False
                case chess.BLACK:
                    mirror = True
            fen = self.chess.fen(mirror)

            # Turn relevant data into indices for NN's embeddings.
            indices = self.tokenizer(fen)
            move_indices = self.chess.move_indices(move, mirror)

            """ TODO:
            - handle score perspective and type
            - transform move_indices to reasonable format
            """ 

            yield {'indices': indices, 'move_indices': move_indices}