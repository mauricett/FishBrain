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

void (*fen_processors[6])() = {
    &process_position,
    &process_turn,
    &process_castling,
    &process_en_passant,
    &process_moves_since,
    &process_moves_total
};


int main()
{
    char fen[] = "4rr1k/pb4q1/1bp2pQp/3p2p1/R7/4P2P/1B2BPP1/3R2K1 w - - 6 32";
    
    int n = 0;
    char *save, *substring;

    for (substring = strtok_r(fen, " ", &save);
         substring != NULL;
         substring = strtok_r(NULL, " ", &save))
    {
        fen_processors[n]();
        ++n;
    }

    return 0;
}
