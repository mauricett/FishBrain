#%%
from torch.utils.data import DataLoader, get_worker_info

from data.dataset import LichessData


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    dataset.config(worker_id)

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
# case 2 - resume=True
# need to load metadata for each worker and properly resume stream
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
