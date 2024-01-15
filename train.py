#%%
from torch.utils.data import DataLoader
from data.dataset import LichessData, worker_init_fn

#%%
lichess_data = LichessData(min_elo=0, resume=False)
dataloader = DataLoader(lichess_data, batch_size=8, 
                        num_workers=2, worker_init_fn=worker_init_fn)


#%%
n = 0
for i in dataloader:
    print(i)
    n += 1
    if n > 4:
        break


#%%
# If python-chess prints an error, this is harmless. The decompressable
# chunks do not cleanly divide pgns. The erroneous data is discarded.
lichess_data = LichessData(min_elo=0, resume=True)
dataloader = DataLoader(lichess_data, batch_size=8, 
                        num_workers=2, worker_init_fn=worker_init_fn)


#%%
n = 0
for i in dataloader:
    print(i)
    n += 1
    if n > 4:
        break
