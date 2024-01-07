#%%
import json

#%%
class Handler:
    """
    Helper to keep the DataStreamer class clean. We keep track of
    which Lichess file we're currently streaming and how deep into
    the file we are. Used to pause and resume training.
    """
    def __init__(self):
        self.file = "training/metadata.json"

    def create_default(self):
        default = dict()
        default['url'] = "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
        default['current_pos'] = "0"
        default['filesize'] = "30047560534"

        with open(self.file, "w") as file:
            json.dump(default, file)

    def load(self):
        try:
            with open(self.file, "r") as file:
                metadata = json.load(file)

        except FileNotFoundError:
            print("Could not find file training/metadata.json,"
                  "creating new default file.")
            self.create_default()
            with open(self.file, "r") as file:
                metadata = json.load(file)   

        finally:
            return(metadata)
