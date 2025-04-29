#include <stdint.h>

enum piece_enc {
	EMPTY  = 0,
	PAWN   = 1,
	KNIGHT = 2,
	BISHOP = 3,
	ROOK   = 4,
	QUEEN  = 5,
	KING   = 6,
	// use the high bit of our 4bit piece representation for color
	WHITE = 8,
	// use highest possible 4bit value as our error code
	ENC_ERROR = 15
};

uint8_t get_high_bits(uint8_t byte);
uint8_t get_low_bits(uint8_t byte);

void set_high_nibble(uint8_t* byte, const uint8_t value);
void set_low_nibble(uint8_t* byte, const uint8_t value);

uint8_t char_to_digit(char digit);
uint8_t get_piece_enc(char piece);
char get_piece_char(uint8_t piece_enc);
