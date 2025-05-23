#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <wchar.h>

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

// PROMOTIONS ????????????

struct GameBitFormat {
	// !!!!!!!!!! should not be part of individual posis !!!!!!!!!!!
	// ****  metadata (6+? bytes) ****
	// 2 bytes elo white
	// 2 bytes elo black
	// 2 bytes num_moves
	// ? bytes header_validator_magic_number?
	//
	// !!!!!!!!!! individual posis !!!!!!!!!!!
	// ****  stuff (4 bytes) ****
	// 6 bits next_move from_square
	// 6 bits next_move to_square
	// 1 bit  is_promotion
	// 1 bit  is_mating_line (did SF find a mating line?) -> changes behaviour of last 16 bits (sf_eval OR mate_in_n)
	// 2 bits outcome condition (checkmate, stalemate, insufficient material, sf eval)
	// 16 bits sf_eval fp16 // mate_in_n
	//
	// **** board position (24 bytes) ****
	// 16 bytes piece_list
	// 8 bytes occupancy_map
	//
	// **** other FEN tokens (1 byte) ****
	// 1 bit turn
	// 4 bits castling
	// 3 bits en passant
	//
	// *** total bytes per game: 6+? + num_moves * 29 bytes
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
