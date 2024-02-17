#%%
import torch
import numpy as np
import chess
import pickle


class Tester:
    def __init__(self, batchsize, tokenizer):
        self.batchsize = batchsize
        self.tokenizer = tokenizer
        self.size_counter = 0
        with open("benchmark/data/sf_test_accuracy.pkl", "rb") as file:
            self.evals = pickle.load(file)

    def __call__(self, model):
        all_predictions = []
        with torch.inference_mode():
            batches = self._batcher()
            for batch in batches:
                nn_input = self._preprocess(batch)
                score = self._call_model(model, nn_input)
                top_moves = self._postprocess(batch, score)
                all_predictions.extend(top_moves)
        return self._calculate_accuracy(all_predictions)
    
    def _legal_moves(self, fen):
        example = []
        board = chess.Board(fen)
        for move in list(board.legal_moves):
            example.append(move.uci())
        return example

    def _batcher(self):
        batch = {'fens': [], 'moves': []}

        n_plies = len(self.evals)
        n_games = len(self.evals[0])

        for ply in range(n_plies):
            for n in range(n_games):
                fen = self.evals[ply][n][0]

                moves = self._legal_moves(fen)
                n_moves = len(moves)

                next_batchsize = n_moves + self.size_counter
                if next_batchsize > self.batchsize:

                    yield batch

                    self.size_counter = 0
                    batch['fens'] = []
                    batch['moves'] = []
                
                batch['fens'].append([fen] * n_moves)
                batch['moves'].append(moves)
                self.size_counter += n_moves
        
        self.size_counter = 0
        yield batch    
    
    def _preprocess(self, batch):
        nn_input = {'fens': [], 'moves': []}

        num_positions = len(batch['fens'])
        for fens, moves in zip(batch['fens'], batch['moves']):

            fen = fens[0].split(" ")
            pos, turn, cas, enp, _, _ = fen
            mirror = turn == 'b'
            
            fen = " ".join([pos, turn, cas, enp])
            fen_token_idx = self.tokenizer.fen(fen, mirror)

            for move in moves:
                move_token_idx = self.tokenizer.move(move, mirror)
                nn_input['fens'].append(fen_token_idx)
                nn_input['moves'].append(move_token_idx)

        nn_input['fens'] = np.array(nn_input['fens'])
        nn_input['moves'] = np.array(nn_input['moves'])
        return nn_input
    
    def _call_model(self, model, nn_input):
        return model(nn_input['fens'], nn_input['moves'])

    def _postprocess(self, batch, scores):
        pointer = 0
        top_moves = []
        n_positions = len(batch['fens'])
        for pos in range(n_positions):

            legal_moves = batch['moves'][pos]
            n_moves = len(legal_moves)

            model_prediction = scores[pointer : pointer+n_moves]            
            pointer += n_moves

            top_score, idx = torch.topk(model_prediction, k=1)
            top_move = legal_moves[idx.cpu()]
            top_moves.append(top_move)
        return top_moves

    def _calculate_accuracy(self, predictions):
        n_plies = len(self.evals)
        n_games = len(self.evals[0])

        results = np.zeros((n_plies, n_games))

        for ply in range(n_plies):
            for game in range(n_games):
                sf_move = self.evals[ply][game][1]
                sf_move = sf_move.uci()

                top_move = predictions[n_games * ply + game]
                if top_move == sf_move:
                    results[ply][game] = 1
        return results