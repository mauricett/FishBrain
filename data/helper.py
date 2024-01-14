import json
import requests
import io

import zstandard as zstd

def transform_worker_id(streamer):
    if streamer.worker_id is None:
        worker_id = ''
    else:
        worker_id = streamer.worker_id
    return worker_id

def save_metadata(streamer):
    worker_id = transform_worker_id(streamer)
    with open("data/metadata" + str(worker_id), "w") as file:
        json.dump(streamer.metadata, file)

def load_metadata(streamer):
    worker_id = transform_worker_id(streamer)
    with open("data/metadata" + str(worker_id), "r") as file:
        json.load(streamer.metadata, file)

def update_metadata(streamer):
    # Keep track of the absolute byte position to support resume.
    worker_id = transform_worker_id(streamer)
    consumed_bytes = streamer.tell()
    if consumed_bytes > streamer.metadata['consumed_bytes']:
        streamer.metadata['consumed_bytes'] = consumed_bytes
        save_metadata(streamer, worker_id)

def download(streamer, start_byte, end_byte='', stream=False):
    url = streamer.metadata['url']
    headers = {"Range" : "bytes=%s-%s" % (start_byte, end_byte)}
    response = requests.get(url, headers=headers, stream=stream)
    return response

def decompress(streamer, response):
    dctx = zstd.ZstdDecompressor()
    sr = dctx.stream_reader(response.raw, read_size=streamer.buffer_size)
    pgn = io.TextIOWrapper(sr)
    return pgn