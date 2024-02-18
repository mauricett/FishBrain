import random
import numpy as np


def sigmoid(x):
    if x < 0:
        return np.exp(x) / (1 + np.exp(x))
    else:
        return 1 / (1 + np.exp(-x))

def scorer(score):
    if score.lstrip("-").isdigit():
        return sigmoid(-float(score) * 0.006)
    else:
        match score:
            case 'C':
                return 1.
            case 'I':
                return 0.5
            case 'S':
                return 0.5

def process_sample(example, tokenizer, scorer):
    fens = example['fens']
    max_ply = len(fens) 
    moves = example['moves']
    scores = example['scores']
    
    rnd = random.randint(min(max_ply-2, 7), max_ply-2)
    fens = fens[rnd]

    pos, turn, _, _ = fens.split(' ')
    if turn == 'b':
        mirror = True
    else:
        mirror = False

    next_idx = rnd + 1
    example['fens'] = tokenizer.fen(fens, mirror)
    example['moves'] = tokenizer.move(moves[next_idx], mirror)
    scores = scores[next_idx]
    example['scores'] = scorer(scores)
    return example