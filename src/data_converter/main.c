#include <stdio.h>
#include <stdint.h>
#include "board_enc.h"
#include "piece_enc.h"


int main() {
	char fen[] = "3k4/bpR5/p3b2q/8/3PQB2/4KP2/PP6/R7";
	Bitboard bitboard = fen_to_bitboard(fen);
	for(int i = 0; i < PIECE_LIST_MAX_BYTES; i++)
		printf("%x", bitboard.piece_list[i]);
	printf("\n");
	print_piece_list(&bitboard);
	return 0;

}
