import json


def transform_worker_id(streamer):
    if streamer.worker_id is None:
        worker_id = ''
    else:
        worker_id = streamer.worker_id
    return worker_id

def save_metadata(streamer):
    worker_id = transform_worker_id(streamer)
    with open("data/metadata/worker" + str(worker_id), "w") as file:
        json.dump(streamer.metadata, file)

def load_metadata(streamer):
    worker_id = transform_worker_id(streamer)
    with open("data/metadata/worker" + str(worker_id), "r") as file:
        streamer.metadata = json.load(file)

def update_metadata(streamer):
    # Keep track of the absolute byte position to support resume.
    worker_id = transform_worker_id(streamer)
    consumed_bytes = streamer.tell()
    if consumed_bytes > streamer.metadata['consumed_bytes']:
        streamer.metadata['consumed_bytes'] = consumed_bytes
        save_metadata(streamer)