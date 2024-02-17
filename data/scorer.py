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