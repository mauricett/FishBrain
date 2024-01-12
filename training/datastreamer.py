#%%
import requests
import json
import io

import zstandard as zstd
import chess.pgn


class DataStreamer:
    def __init__(self, metadata:    dict, 
                       resume:      bool = False, 
                       buffer_size: int  = 16 * 10**6):
        # metadata holds the data url and the number of bytes
        # already consumed to allow resuming a stream.
        self.metadata = metadata
        self.buffer_size = buffer_size
        # resume_stream() will restart the stream only approximately
        # at the location where we stopped, due to technical reasons.
        if resume:
            self.response, self.pgn, self.start_byte = self.resume_stream()
        if not resume:
            self.response, self.pgn = self.start_stream()
            self.start_byte = 0

    def start_stream(self, start_byte=0):
        headers = {"Range" : "bytes=%s-" % start_byte}
        response = requests.get(url, headers=headers, stream=True)
        dctx = zstd.ZstdDecompressor()
        sr = dctx.stream_reader(response.raw, read_size=self.buffer_size)
        pgn = io.TextIOWrapper(sr)
        return response, pgn

    def resume_stream(self):
        print("Resuming stream. Finding entry point for decompression.")
        last_byte = self.metadata['last_byte']
        frame_test_byte = last_byte + 10 * 10**6
        headers = {"Range" : "bytes=%s-%s" % (last_byte, frame_test_byte)}
        response = requests.get(url, headers=headers)
        content = response.content
        response.close()
        frame_start = self.find_frame_start(content)
        start_byte = last_byte + frame_start
        response, pgn = self.start_stream(start_byte=start_byte)
        # Discard first game, because the decompressable chunks do not
        # cleanly divide the pgns into legal game data.
        try:
            print("If you receive an error message from python-chess, "
                  "this is harmless and expected. Ignore.")
            print("-------------")
            _ = chess.pgn.read_game(pgn)
            print("-------------")
        finally:
            return response, pgn, start_byte

    def find_frame_start(self, content):
        # Finds byte position of first decompressable frame.
        FRAME_MAGIC_NUMBER = int('0xFD2FB528', 16)
        for n in range(len(content)):
            number = int.from_bytes(content[n:n+4], byteorder='little')
            if number == FRAME_MAGIC_NUMBER:
                return n
    
    def tell(self):
        # Returns the absolute number of bytes consumed.
        # response.raw.tell() returns the bytes relative
        # to where we restarted the stream. -> Add start_byte.
        return self.response.raw.tell() + self.start_byte
    
    def store_metadata(self):
        with open("training/metadata.json", "w") as file:
            json.dump(self.metadata, file)

    def read_game(self):
        # Keep track of the absolute byte position to support resume.
        last_byte = self.tell()
        if last_byte > self.metadata['last_byte']:
            self.metadata['last_byte'] = last_byte
            self.store_metadata()
        # Find a pgn which contains evals.
        try:
            while True:
                game = chess.pgn.read_game(self.pgn)
                has_eval = game.next().eval()
                if not has_eval:
                    continue
                return game
        except:
            return self.read_game()