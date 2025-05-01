#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>


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

uint8_t get_high_bits(uint8_t byte) {
	// shift high bits into the low nibble
	return byte >> 4;
}

uint8_t get_low_bits(uint8_t byte)  {
	// return with cleared high bits
	return byte & 0x0F;
}

void set_high_nibble(uint8_t* byte, const uint8_t value) { 
	// clear and set high nibble
	*byte &= 0x0F;
	*byte |= (value << 4);
}

void set_low_nibble(uint8_t* byte, const uint8_t value)  { 
	// clear and set low nibble
	*byte &= 0xF0;
	*byte |= value;
}

uint8_t char_to_digit(char digit) {
	return (uint8_t) (digit - '0');
}

uint8_t get_piece_enc(char piece) {
	uint8_t enc = EMPTY;
	if (isupper(piece))
		enc += WHITE;
	switch(tolower(piece)) {
		case 'p': return enc+PAWN;
		case 'n': return enc+KNIGHT;
		case 'b': return enc+BISHOP;
		case 'r': return enc+ROOK;
		case 'q': return enc+QUEEN;
		case 'k': return enc+KING;
		default:  return ENC_ERROR; 
	}
}

char get_piece_char(uint8_t piece_enc) {
	bool is_white = (piece_enc > WHITE) && (piece_enc < ENC_ERROR);
	if (is_white)
		piece_enc -= WHITE;
	switch(piece_enc) {
		case EMPTY:  return '.';
		case PAWN:   return is_white ? 'P' : 'p';
		case KNIGHT: return is_white ? 'N' : 'n';
		case BISHOP: return is_white ? 'B' : 'b';
		case ROOK:   return is_white ? 'R' : 'r';
		case QUEEN:  return is_white ? 'Q' : 'q';
		case KING:   return is_white ? 'K' : 'k';
		default:     return '?';
	}
}

