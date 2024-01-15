#%%
from torch.utils.data import DataLoader
from data.dataset import LichessData, worker_init_fn

#%%
lichess_data = LichessData(min_elo=2000, resume=False)

dataloader = DataLoader(lichess_data, batch_size=128, 
                        num_workers=5, worker_init_fn=worker_init_fn)


#%%
n = 0
for i in dataloader:
    print(i)
    n += 1
    if n > 12:
        break