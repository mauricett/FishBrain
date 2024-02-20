

#### 2024.02.20
### Managing model data / training runs:
- Have a helper class to organize models and training data?

### Training script:
- Add proper resume function
- Add arguments for mixed precision training:
    - enable=True/False
    - arguments for GradScaler
- Save GradScaler state
- Save more infos in model_dict, such as
    - batchsize for each checkpoint
    - 

### DataLoader:
- Add support for old DataLoader

### Some basic ease-of-use extensions to the scripts:
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
