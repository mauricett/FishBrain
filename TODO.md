

#### 2024.02.20
### Some basic ease-of-use extensions to the scripts:
- Merge all training scripts into single script.
- In speedtest.py and hf_trainer(_multi).py, add:
    - argparse launch args for 
        - CPU or CUDA + optional device IDs
        - batchsize
        - num workers
    - print timestamp of run and params used for easier eyeballing and separation of runs
    - optional logging of ^ and what not
    - 

### Hyperparam search/comparisons for optimal training perf:
- Add code to find best combined values for batchsize, device ids (1-vs-m), num_workers
- Add code to compare and find best activation function wrt model accuracy, maybe also check best convergence rates?
- Test relation between width, depth and speed. Best combo for tree search?
- BatchNorm instead of LayerNorm?

### Open problems
- Predict scores for all legal moves at once.
- Cleaner embedding style for convolutional architecture.
- Bitboards to get legal moves.
- Tree search (maybe slow one in Python first is ok?)
- Test standard fully-connected transformer on longer training run?
- 

### Other things to test:
- 
