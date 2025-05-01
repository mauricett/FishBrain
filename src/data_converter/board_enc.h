#include <stdint.h>


#define PIECE_LIST_MAX_BYTES 16

typedef struct bitboard_struct {
	uint64_t occupancy_map;
	// max 32 pieces, 4 bits per piece
	uint8_t  piece_list[16];
	uint8_t  num_pieces;
} Bitboard;


Bitboard fen_to_bitboard(char* board_string);
void print_piece_list(Bitboard* bitboard);

