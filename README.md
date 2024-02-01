# FishBrain

<div align="center">
    <img src="img/logo_alpha_4.png" alt="Logo" width="256" height="256">
</div>

&nbsp;
&nbsp;

<div align="center">
FishBrain is a chess engine which uses a neural network trained on Stockfish evaluations.
</div>

&nbsp;
&nbsp;
# High-Quality Dataset Soon
Currently, FishBrain streams training data directly from Lichess, bypassing the need to save large files on disk. However, this has proven to be too slow. I am currently curating a dataset from all Lichess games that have been evaluated by Stockfish 16. This dataset will be published soon.

# Technical Reports Soon
I will document my experiments and results in technical reports.

# First Practice Run
I have begun a first training run, and the results are promising. I measure performance by counting how often FishBrain predicts the top move of SF16. For this purpose, I have created a dataset that measures the performance at different numbers of half-moves.
The results are in the following figure, where I compare the accuracy of FishBrain to SF16 (without tree search) and a random agent. We're already better than SF16 without search!

<div align="center">
    <img src="benchmark/sf_0node_accuracy.png" alt="FishBrain Benchmark Results">
</div>

I'm heavily bottlenecked by the live-streamed DataLoader right now, but that problem is going to be solved soon.

# Feedback and discussions
I'm happy about any feedback or discussion. Feel free to reach out.
