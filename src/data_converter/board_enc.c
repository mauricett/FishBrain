#include <stdio.h>
#include <ctype.h>
#include <stdint.h>
#include "board_enc.h"
#include "piece_enc.h"


void fill_occupancy_map(Bitboard* bitboard, char next_square);
void fill_piece_list(Bitboard* bitboard, char next_square);
void add_empty_squares(uint64_t* occupancy_map, uint8_t num_empty_squares);
void add_occupied_square(uint64_t* occupancy_map);

Bitboard fen_to_bitboard(char* board_string) {
	Bitboard bitboard = {0};
	// loop through null-terminated string that holds the board
	while (*board_string) {
		fill_occupancy_map(&bitboard, *board_string);
		if isalpha(*board_string) 
			fill_piece_list(&bitboard, *board_string);
		board_string++;
	}
	return bitboard;
}

void fill_occupancy_map(Bitboard* bitboard, char next_square) {
	if isdigit(next_square)
		add_empty_squares(&bitboard->occupancy_map, char_to_digit(next_square));
	else if isalpha(next_square)
		add_occupied_square(&bitboard->occupancy_map);
}

void fill_piece_list(Bitboard* bitboard, char next_square) {
	uint8_t piece_enc = get_piece_enc(next_square);
	uint8_t piece_idx = bitboard->num_pieces;
	// two pieces per byte, get correct byte to write to
	int byte_idx = piece_idx / 2;
	uint8_t* target_byte = &bitboard->piece_list[byte_idx];
	// if piece index is even -> write to high bits
	if (piece_idx % 2 == 0)
		set_high_bits(target_byte, piece_enc);
	else
		set_low_bits(target_byte, piece_enc);
	bitboard->num_pieces++;
}

void add_empty_squares(uint64_t* occupancy_map, uint8_t num_empty_squares) {
	// zero-bits represent empty squares
	*occupancy_map = *occupancy_map << num_empty_squares;
}

void add_occupied_square(uint64_t* occupancy_map) {
	// shift to make place for next piece, then activate low bit
	*occupancy_map = (*occupancy_map << 1) + 1;
}

void print_piece_list(Bitboard* bitboard) {
	char piece_char;
	uint8_t piece_enc;
	uint8_t* piece_list = bitboard->piece_list;
	// loop through piece_list, get pieces at low and high bits
	for (int byte = 0; byte < PIECE_LIST_MAX_BYTES; byte++) {
		piece_enc = get_high_bits(*piece_list);
		piece_char = get_piece_char(piece_enc);
		putchar(piece_char);
		piece_enc = get_low_bits(*piece_list);
		piece_char = get_piece_char(piece_enc);
		putchar(piece_char);
		piece_list++;
	}
}
