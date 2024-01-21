"""
    THIS IS A MOCK INTERFACE, NOT IMPLEMENTED YET :)
"""

import inference.fishbrain as fishbrain

engine = fishbrain.Engine(
    model="model/fishweights.pt",
    device='cuda',
    tree_search=True,
    node_limit=10**4
)

# The engine takes positions in the FEN format. Example position:
fen = "rnb2rk1/pp2bppp/2p1p3/q7/3P4/1BN1P3/PPP2PPP/R1BQK2R w KQ - 2 12"
score = engine.eval(fen)