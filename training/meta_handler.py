#%%
import json

#%%
class MetaHandler:
    """
    Helper to keep the DataStreamer class clean. We keep track of
    which Lichess file we're currently streaming and how deep into
    the file we are. Used to pause and resume training.
    """
    def __init__(self):
        self.file = "training/metadata.json"
        self.metadata: dict = self.load_metadata()

    def create_default(self):
        default = dict()
        default['url'] = "lichess.org"
        default['frame_pos'] = "12345"
        default['filesize'] = "124567"
        with open(self.file, "w") as file:
            json.dump(default, file)

    def load_metadata(self):
        try:
            with open(self.file, "r") as file:
                metadata = json.load(file)
        except FileNotFoundError:
            print("Could not find file training/metadata.json, "
                  "creating default file.")
            self.create_default()
            with open(self.file, "r") as file:
                metadata = json.load(file)
        finally:
            self.print_metadata(metadata)
            return(metadata)
    
    def print_metadata(self, metadata):
        print("Current data file: %s" % metadata['url'])
        print("Selected frame is at byte %s." % metadata['frame_pos'])
        print("Total file size: %s bytes" % metadata['filesize'])


#%%
meta_handler = MetaHandler()

#%%

