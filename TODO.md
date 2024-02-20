

#### 2024.02.20
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

### Other things to test:
- 
