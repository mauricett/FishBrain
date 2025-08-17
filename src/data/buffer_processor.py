import io
from ctypes import POINTER, c_char



class BufferProcessor:
    def __init__(self, **kwargs):
        self.n_buffered_games = 0
        self.n_saved_games = 0

        self.n_moves = []
        self.outcomes = []
        self.moves_io = io.BytesIO()
        self.evals_io = io.BytesIO()

        self.n_max_games = kwargs["games_per_file"]
        self.move_sep    = kwargs["move_separator"]
        self.game_sep    = kwargs["game_separator"]
        self.c_processor = kwargs["c_processor"]
        #self.c_argtypes = c_argtypes


    @property
    def is_full(self):
        return self.n_buffered_games == self.n_max_games


    @property
    def is_empty(self):
        return self.n_buffered_games == 0


    def clear(self):
        self.n_moves = []
        self.outcomes = []
        self.moves_io = io.BytesIO()
        self.evals_io = io.BytesIO()
        self.n_buffered_games = 0


    def add_game(self, data):
        self.n_moves.append(data.n_moves)
        self.outcomes.append(data.outcome)
        moves  = self.move_sep.join(data.moves) + self.game_sep
        evals  = self.move_sep.join(data.evals) + self.game_sep
        self.evals_io.write(evals.encode('UTF-8'))
        self.moves_io.write(moves.encode('UTF-8'))
        self.n_buffered_games += 1


    def process_and_clear(self):
        try:
            m_buf = self.moves_io.getbuffer()
            e_buf = self.evals_io.getbuffer()
            self.c_processor(
                (c_char * len(m_buf)).from_buffer(m_buf),
                (c_char * len(e_buf)).from_buffer(e_buf),
            )
            self.n_saved_games += self.n_buffered_games
        except Exception as e:
            print(f"Skipping chunk due to exception in C processor: {e}")
        finally:
            # no recovery if exception, just clear buffer and keep going
            self.clear()
