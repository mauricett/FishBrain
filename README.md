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
The goal of this project is to explore small but powerful neural network architectures that excel at chess. 
In contrast, Stockfish uses a novel NN architecture which sacrifices capacity for extreme speed.
Leela makes no such sacrifice and instead uses a large, powerful but slow NN.
FishBrain explores the middle ground - NN architectures with no architectural sacrifices but small model size, which makes it significantly faster to run and train compared to Leela.

&nbsp;
&nbsp;
I believe this is a viable approach to create a competitive chess engine at home, once we also do tree search.

# FishBrain v0
I finished the first NN (see v0_legacy) in 2024 and it works okay. The NN is much smaller than DeepMind's and achieves a Blitz Elo of about 1800. The code is "research quality", it sucks. Let's create a great codebase for v1.

# Dataset
Old dataset: <a href="https://huggingface.co/datasets/mauricett/lichess_sf">HuggingFace dataset</a>.
Shitty format, I will rework. It's curated Lichess data from 2023. New dataset will be Jan 2023 - June 2025 and come in a friendlier format.

# Future directions?
<a href="https://github.com/LeelaChessZero">LeelaChessZero</a> has produced enormous amount of data which is freely available. Would be cool to utilize this in the future, on top of the Stockfish data. <a href="https://storage.lczero.org/files/">So much data...</a>
