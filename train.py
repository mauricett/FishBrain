#%%
from torch.utils.data import DataLoader
from data.dataset import LichessData, worker_init_fn

#%%
lichess_data = LichessData(min_elo=1000, resume=False)
# !!! Use num_workers <= 5 to prevent being rate-limited by Lichess !!!
dataloader = DataLoader(lichess_data, batch_size=128, 
                        num_workers=4, worker_init_fn=worker_init_fn)


#%%
n = 0
for i in dataloader:
    print(i)
    n += 1
    if n > 12:
            break

#%%
