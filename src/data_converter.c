#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define MAX_FEN_PROCESSORS 6

void process_position() {};
void process_turn() {};
void process_castling() {};
void process_en_passant() {};
void process_moves_since() {};
void process_moves_total() {};

void (*fen_processors[MAX_FEN_PROCESSORS])() = {
    &process_position,
    &process_turn,
    &process_castling,
    &process_en_passant,
    &process_moves_since,
    &process_moves_total
};

struct GameBitFormat {
	// metadata
	//
	// ****  2 bytes (16 bits) ****
	// 1 bit  is_final_move (i.e. next data will be new game)
	// 1 bit  stockfish eval (0) or end condition (1)?
	// 6 bits next_move from_square
	// 6 bits next_move to_square
	//     -> 2 bits free
	//
	// **** board position 24 bytes ****
	// 16 bytes piece_list
	// 8 bytes occupancy_map
	//
	// **** other FEN tokens 4 bytes ****
	// 1 bit turn
	// 4 bits castling
	// 3 bits en passant
	// 8 bits moves since
	// 16 bits total moves
	//
	// **** outcome 2 bytes ****
	// 16 bits sf eval fp16 or end_condition char
	//
	// ??????????????????????????????
	// which "sequence information" do i need?
	// morally, we use individual FEN positions for training,
	// but our data is sequences of positions...
	// what data would get lost if i stored individual positions?
	//
	// ??????????????????????????????
	// do we exclude games with certain end conditions, e.g. repetition?
};

char fen[] = "4rr1k/pb4q1/1bp2pQp/3p2p1/R7/4P2P/1B2BPP1/3R2K1 w - - 6 32";

int main()
{
    int n = 0;
    char *save, *substring;

    for (
        substring = strtok_r(fen, " ", &save);
        substring != NULL;
        substring = strtok_r(NULL, " ", &save)
    ) {
		printf("%s\n", substring);
        fen_processors[n]();
        ++n;
    }

	//struct GameBitFormat data = {0};

    return 0;
}
