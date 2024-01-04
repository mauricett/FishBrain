#%%
import requests
import chess
import chess.pgn
import zstandard as zstd
import io

#%%
# we can automatically retrieve the download links from lichess.
# not sure if i wanna use this.
url = "https://database.lichess.org/standard/list.txt"
response = requests.get(url, stream=True)
for line in response.iter_lines():
    print(line)
    break

#%%
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"

headers = {"Range":"bytes=0-40000"}
response = requests.get(url, headers=headers, stream=True)
iterator = response.iter_content(chunk_size=40001)

for chunk in iterator:
    data = chunk

#%%
dctx = zstd.ZstdDecompressor()
# this works, but I wanna see if I can skip frames and decompress individually 
dctx.stream_reader(data[:40000]).read(1024)

#%%
# this is zstandard's magic number for a skippable frame 
int.from_bytes(data[0:4], byteorder='little') == int('0x184D2A50', 16)
# size of user data, here 4 bytes
int.from_bytes(data[4:8], byteorder='little') == 4

#%%
# after skipping the 4 bytes of user data, we find the next frame.
# it's not a "skippable frame" this time, 
# but the magic number corresponds to a general zstd frame.
int.from_bytes(data[12:16], byteorder='little') == int('0xFD2FB528', 16)

#%%



#%%