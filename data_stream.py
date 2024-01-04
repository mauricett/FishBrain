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

headers = {"Range":"bytes=0-400000000"}
response = requests.get(url, headers=headers, stream=True)
iterator = response.iter_content(chunk_size=400000001)

for chunk in iterator:
    data = chunk

#%%
dctx = zstd.ZstdDecompressor()
# this works, but I wanna see if I can skip frames and decompress individually 
dctx.stream_reader(data[:]).read(1024)

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
# the frame header descriptor equals 4, i.e. 0000100 in binary.
# this means, that the frame header contains only the content_checksum_flag?
int.from_bytes(data[16:17], byteorder='little') == 4
# additional flags
int.from_bytes(data[17:18], byteorder='little')


# 4 bytes for magic number, 2 bytes for frame header?
# I think 1 byte for frame header descriptor
# and 1 byte for additional flags (see above)
zstd.frame_header_size(data) == 6

#%%
# this should be the block header
block_header = int.from_bytes(data[18:21], byteorder='little')
block_size = block_header >> 3

#%%
def traverse_blocks(data, block_header_pos):
    header = data[block_header_pos : block_header_pos + 3]
    header = int.from_bytes(header, byteorder='little')
    is_last = bin(header)[-1]
    block_content_size = header >> 3
    next_block_pos = block_header_pos + 3 + block_content_size
    return int(is_last), block_content_size, next_block_pos

#%%
is_last = 0
next_block_pos = 18
total_size = 0
while not is_last:
    is_last, block_content_size, next_block_pos = traverse_blocks(data, next_block_pos)
    print("Size: " + str(block_content_size))
    print("Is last: " + str(is_last))
    total_size += block_content_size + 3

#%%
checksum_bytes = 4
next_frame_pos = next_block_pos + checksum_bytes

#%%
# next frame magic number
# is another skippable frame
int.from_bytes(data[next_frame_pos : next_frame_pos + 4], byteorder='little')

#%%
# care, some frames have checksums, while others don't
zstd.get_frame_parameters(data[12:]).has_checksum == True
zstd.get_frame_parameters(data[next_frame_pos:]).has_checksum == False