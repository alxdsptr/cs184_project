#include <string>
#include <vector>

void decompressBC1Block(const uint8_t* block, uint8_t out[4][4][4]);

bool decompressDDS(const std::string& path,
                          std::vector<unsigned char>& outPixels,
                          int& outWidth, int& outHeight);
                