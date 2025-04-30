#include <stdint.h>


#define PIECE_LIST_MAX_BYTES 16

typedef struct bitboard_struct {
	uint64_t occupancy_map;
	uint8_t  piece_list[16]; // max 32 pieces, 4 bits per piece
	uint8_t  num_pieces;
} Bitboard;


Bitboard fen_to_bitboard(char* board_string);

void print_piece_list(Bitboard* bitboard);

void fill_occupancy_map(Bitboard* bitboard, char next_square);
void fill_piece_list(Bitboard* bitboard, char next_square);

void add_empty_squares(uint64_t* occupancy_map, uint8_t num_empty_squares);
void add_occupied_square(uint64_t* occupancy_map);
