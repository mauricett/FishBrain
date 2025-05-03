#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>


void process_position() {};
void process_turn() {};
void process_castling() {};
void process_en_passant() {};
void process_moves_since() {};
void process_moves_total() {};

int main() {

	void (*fen_processors[6])() = {
		&process_position,
		&process_turn,
		&process_castling,
		&process_en_passant,
		&process_moves_since,
		&process_moves_total
	};

	char fen[] = "4rr1k/pb4q1/1bp2pQp/3p2p1/R7/4P2P/1B2BPP1/3R2K1 w - - 6 32";
	
	char *delim = " ";
	char *save;
	
	/*
	// **************************************************
	// variant 1
	int processed_strings = 0;
	char *substring = strtok_r(fen, delim, &save);
	while (substring)
	{
		fen_processors[processed_strings]();
		++processed_strings;
		substring = strtok_r(NULL, delim, &save);
	}
	

	// **************************************************
	// variant 2
	char *substring = strtok_r(fen, delim, &save);
	for (int n_str = 0; n_str < 6; ++n_str)
	{
		fen_processors[n_str]();
		substring = strtok_r(NULL, delim, &save);
	}
	*/

	// **************************************************
	// variant 3
	int n = 0;
	char *substring;
	for (substring = strtok_r(fen, delim, &save);
		 substring != NULL;
		 ++n, substring = strtok_r(NULL, delim, &save))
	{
		fen_processors[n]();
	}

	return 0;
}
