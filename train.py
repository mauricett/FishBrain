#%%
from torch.utils.data import DataLoader
from data.dataset import LichessData, worker_init_fn

#%%
lichess_data = LichessData(min_elo=1000, resume=False)

# !!! Use num_workers <= 5 to prevent being rate-limited by Lichess !!!
dataloader = DataLoader(lichess_data, batch_size=8, 
                        num_workers=1, worker_init_fn=worker_init_fn)


#%%
n = 0
batches = []
for i in dataloader:
    batches.append(i)
    n += 1
    if n > 8:
            break


#%%
bth_batch = 3
ith_sample = 0

b = bth_batch
i = ith_sample

print("\n___piece positions___")
print(batches[b]['indices'][i][0:64].view(8, 8))
print("\n___color positions___")
print(batches[b]['indices'][i][64:128].view(8, 8))
print("\n___castling rights___")
print(batches[b]['indices'][i][128:132])
print("\n___en passent___")
print(batches[b]['indices'][i][132:133])
print("\n___move___")
print(batches[b]['indices'][i][133:135])
print("\n___score___")
print(batches[b]['score'][i])