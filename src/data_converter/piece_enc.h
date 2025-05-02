#include <stdint.h>

uint8_t get_high_bits(uint8_t byte);
uint8_t get_low_bits(uint8_t byte);

void set_high_bits(uint8_t* byte, const uint8_t value);
void set_low_bits(uint8_t* byte, const uint8_t value);

uint8_t char_to_digit(char digit);
uint8_t get_piece_enc(char piece);
char get_piece_char(uint8_t piece_enc);
