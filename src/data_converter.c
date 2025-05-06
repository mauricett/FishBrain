#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#define MAX_FEN_PROCESSORS 6

void process_position() {};
void process_turn() {};
void process_castling() {};
void process_en_passant() {};
void process_half_moves() {};
void process_full_moves() {};

void (*fen_processors[MAX_FEN_PROCESSORS])() = {
    &process_position,
    &process_turn,
    &process_castling,
    &process_en_passant,
    &process_half_moves,
    &process_full_moves
};

struct GameBitFormat {
	// !!!!!!!!!! should not be part of individual posis !!!!!!!!!!!
	// ****  metadata (6 bytes) ****
	// 2 bytes elo white
	// 2 bytes elo black
	// 2 bytes num_moves
	//
	// !!!!!!!!!! individual posis !!!!!!!!!!!
	// ****  stuff (4 bytes) ****
	// 1 bit  is_final_move (i.e. next data will be new game)
	// 1 bit  free bit (is_check?)
	// 6 bits next_move from_square
	// 6 bits next_move to_square
	// 2 bits outcome condition (checkmate, stalemate, insufficient material, sf eval)
	// 16 bits sf eval fp16
	//
	// **** board position (24 bytes) ****
	// 16 bytes piece_list
	// 8 bytes occupancy_map
	//
	// **** other FEN tokens (4 bytes) ****
	// 1 bit turn
	// 4 bits castling
	// 3 bits en passant
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
