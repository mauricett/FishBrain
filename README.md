# FishBrain

<div align="center">
    <img src="img/logo_alpha.png" alt="Logo" width="196" height="196">
</div>

<div align="center">
Project: training NNs on Stockfish data.
</div>

&nbsp;
&nbsp;

# FishBrain v1 - THE BIG REBOOT (YES!!!)
The goal of this project is to explore small NN architectures which excel at chess.
In contrast, <a href="https://github.com/official-stockfish/Stockfish">Stockfish</a> uses an NN architecture which sacrifices capacity for extreme speed.
<a href="https://github.com/LeelaChessZero">Leela</a> uses a powerful but slow NN, which is infeasible to train on consumer hardware.

&nbsp;
&nbsp;
FishBrain explores the middle ground - NNs with no architectural sacrifices but small model size.
I believe this is a viable approach to create a competitive chess engine at home, once we also do tree search.
More infos on FishBrain's special architecture and training soon.

# FishBrain v0
I finished the first NN (see v0_legacy) in 2024 and it works okay. The NN is much smaller than <a href="https://arxiv.org/html/2402.04494v1">DeepMind's</a> and achieves a Blitz Elo of about 1800. The code is "research quality", it sucks, sorry. Let's create a great codebase for v1.

# Dataset
I am in the process of updating and reworking the dataset. It's going to be 2.5x bigger and a friendlier format.

&nbsp;
&nbsp;
Old dataset: <a href="https://huggingface.co/datasets/mauricett/lichess_sf">HuggingFace dataset</a>.
The data is extracted from the <a href="https://database.lichess.org/">lichess.org open database</a> and contains literally all games from 2023 for which Stockfish evaluations were available. The format is a bit shittye. (It uses on-the-fly decompression with <a href="https://github.com/facebook/zstd">Zstandard</a>.

# Future directions?
has produced enormous amount of data which is freely available. Would be cool to utilize this in the future, on top of the Stockfish data. <a href="https://storage.lczero.org/files/">So much data...</a>
