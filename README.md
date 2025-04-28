# FishBrain

<div align="center">
    <img src="img/logo_alpha.png" alt="Logo" width="128" height="128">
</div>

&nbsp;

<div align="center">
Project: training NNs on Stockfish data.
</div>

&nbsp;
&nbsp;
# Dataset
Old dataset: <a href="https://huggingface.co/datasets/mauricett/lichess_sf">HuggingFace dataset</a>.
Shitty format, I will rework. It's curated Lichess data from 2023. New dataset will be Jan 2023 - June 2025 and come in a friendlier format.

# FishBrain v0
I finished the first NN (see v0_legacy) in 2024 and it works okay. The NN is much smaller than DeepMind's and achieves a Blitz Elo of about 1800. The code is "research quality", it sucks. Let's create a great codebase for v1.

# Future directions?
<a href="https://github.com/LeelaChessZero">LeelaChessZero</a> has produced enormous amount of data which is freely available. Would be cool to utilize this in the future, on top of the Stockfish data. <a href="https://storage.lczero.org/files/">So much data...</a>
