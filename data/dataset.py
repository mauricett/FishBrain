#%%
import requests
import json

import chess.pgn
import torch
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
            # Update number of bytes consumed by streamer.
            update_metadata(self.data_streamer)
            return game

    def calculate_score(self):
        # Get SF eval, returns None if no eval available.
        score = self.chess.next_eval()

        # If there's an SF eval, apply transforms and return result.
        if score is not None:
            # Gets integer value in units of centipawns.
            score = score.relative.score(mate_score=1000000)
            # Change perspective, always needed here.
            score = -score
            # Scale score by small value that doesn't saturate sigmoid.
            return torch.sigmoid(torch.tensor(score) * 0.006)

        # If there's no SF eval, use terminal outcome to assign score.
        else:
            next_pos = self.chess.game.next()
            outcome = next_pos.board().outcome()
                
            # If no terminal condition found, return None to skip game.
            if outcome is None:
                return None

            # If a terminal condition exists, use it to assign score.
            else:
                outcome = outcome.termination
                # We only assign scores for these three outcomes.
                if outcome == chess.Termination.CHECKMATE:
                    return torch.tensor(1.)
                elif outcome == chess.Termination.STALEMATE:
                    return torch.tensor(0.5)
                elif outcome == chess.Termination.INSUFFICIENT_MATERIAL:
                    return torch.tensor(0.5)
                # Skip game if outcome is neither of the above.
                else:
                    return None

    def __iter__(self):
        # Use while...yield to turn streamer into generator.
        while True:
            # Get next game from streamer and pick random position.
            self.chess.game = self.read_game()
            self.chess.game = self.chess.random_ply()

            # Get SF eval or handle terminal position.
            score = self.calculate_score()
            if score is None:
                continue

            move = self.chess.next_move()
            # If black's turn, use a mirrored and color-flipped board.
            # This makes it so every position looks like it was white's
            # perspective and turn.
            mirror = self.chess.turn() == chess.BLACK

            fen = self.chess.fen(mirror)
            fen_indices = self.tokenizer.fen(fen)
            move_indices = self.tokenizer.move(move, mirror)

            indices = torch.cat([fen_indices, move_indices])
            yield {'indices': indices, 'score': score}