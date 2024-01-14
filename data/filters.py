def has_evals(game):
    has_eval = game.next().eval()
    return has_eval

def min_elos(game, min_elo):
    elo_w = int(game.headers['WhiteElo'])
    elo_b = int(game.headers['BlackElo'])
    elo_check = (elo_w > min_elo) and (elo_b > min_elo)
    return elo_check