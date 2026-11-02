#pragma once
#include <cstddef>

// Zeroed memory for big random-access tables (the transposition table), on large pages when the OS
// allows it: 2 MB pages instead of 4 KB mean far fewer address-translation misses on random probes.
// Windows needs the "Lock pages in memory" right for the account; without it this falls back to
// normal pages. Linux uses transparent huge pages (no special rights).
void* allocLargePages(size_t bytes, bool& large);
void freeLargePages(void* p);
