#pragma once
#include <string>

// Universal Chess Interface loop on stdin/stdout, so the engine can play in other GUIs and matches.
void uciLoop(const std::string& tbDir, const std::string& bookPath);
