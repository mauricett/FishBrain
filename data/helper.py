import json
import requests
import io

import zstandard as zstd


def save_metadata(streamer):
    with open("data/metadata" + str(streamer.worker_id), "w") as file:
        json.dump(streamer.metadata, file)

def load_metadata(streamer):
    with open("data/metadata" + str(streamer.worker_id), "r") as file:
        json.load(streamer.metadata, file)

def update_metadata(streamer):
    # Keep track of the absolute byte position to support resume.
    consumed_bytes = streamer.tell()
    if last_byte > streamer.metadata['consumed_bytes']:
        streamer.metadata['consumed_bytes'] = consumed_bytes
        save_metadata(streamer, streamer.worker_id)

def download(streamer, start_byte: str, end_byte: str, stream: bool):
    url = streamer.metadata['url']
    headers = {"Range" : "bytes=%s-%s" % (start_byte, end_byte)}
    response = requests.get(url, headers=headers, stream=stream)
    return response

def decompress(streamer, response):
    dctx = zstd.ZstdDecompressor()
    sr = dctx.stream_reader(response.raw, read_size=streamer.buffer_size)
    pgn = io.TextIOWrapper(sr)
    return pgn