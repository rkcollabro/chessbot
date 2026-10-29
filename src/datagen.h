#pragma once
#include <string>
#include <vector>

int buildBook(const std::string& out, int maxPly, const std::vector<std::string>& pgns);
int extractPositions(const std::string& out, int perGame, int maxPositions, const std::vector<std::string>& pgns);
int tune(const std::string& dataFile, const std::string& outInc, int epochs);
