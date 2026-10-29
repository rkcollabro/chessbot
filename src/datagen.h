#pragma once
#include <string>
#include <vector>

int buildBook(const std::string& outTxt, int maxPly, int minElo, int minBaseSeconds,
              const std::vector<std::string>& pgns);
int bookFromText(const std::string& txt, const std::string& outBin);
int extractPositions(const std::string& out, int perGame, int maxPositions, const std::vector<std::string>& pgns);
int tune(const std::string& dataFile, const std::string& outInc, int epochs);
