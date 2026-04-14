#include "utils.h"
#include <gli/gli.hpp>
#include <gli/convert.hpp>

// ── BC1 (DXT1) software decompression ─────────────────────────
void decompressBC1Block(const uint8_t* block, uint8_t out[4][4][4]) {
    uint16_t c0 = block[0] | (block[1] << 8);
    uint16_t c1 = block[2] | (block[3] << 8);

    uint8_t colors[4][4]; // [index][rgba]
    // Decode 5-6-5 to 8-8-8
    colors[0][0] = (uint8_t)(((c0 >> 11) & 0x1F) * 255 / 31);
    colors[0][1] = (uint8_t)(((c0 >> 5)  & 0x3F) * 255 / 63);
    colors[0][2] = (uint8_t)(( c0        & 0x1F) * 255 / 31);
    colors[0][3] = 255;

    colors[1][0] = (uint8_t)(((c1 >> 11) & 0x1F) * 255 / 31);
    colors[1][1] = (uint8_t)(((c1 >> 5)  & 0x3F) * 255 / 63);
    colors[1][2] = (uint8_t)(( c1        & 0x1F) * 255 / 31);
    colors[1][3] = 255;

    if (c0 > c1) {
        for (int k = 0; k < 3; k++) {
            colors[2][k] = (uint8_t)((2 * colors[0][k] + colors[1][k] + 1) / 3);
            colors[3][k] = (uint8_t)((colors[0][k] + 2 * colors[1][k] + 1) / 3);
        }
        colors[2][3] = 255;
        colors[3][3] = 255;
    } else {
        for (int k = 0; k < 3; k++) {
            colors[2][k] = (uint8_t)((colors[0][k] + colors[1][k]) / 2);
        }
        colors[2][3] = 255;
        colors[3][0] = colors[3][1] = colors[3][2] = 0;
        colors[3][3] = 0; // transparent black
    }

    uint32_t indices = block[4] | (block[5] << 8) | (block[6] << 16) | (block[7] << 24);
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int idx = indices & 0x3;
            indices >>= 2;
            out[row][col][0] = colors[idx][0];
            out[row][col][1] = colors[idx][1];
            out[row][col][2] = colors[idx][2];
            out[row][col][3] = colors[idx][3];
        }
    }
}

// ── BC3 (DXT5) software decompression ─────────────────────────
void decompressBC3Block(const uint8_t* block, uint8_t out[4][4][4]) {
    // First 8 bytes: alpha block
    uint8_t a0 = block[0];
    uint8_t a1 = block[1];
    uint8_t alphas[8];
    alphas[0] = a0;
    alphas[1] = a1;
    if (a0 > a1) {
        for (int i = 1; i <= 6; i++)
            alphas[i + 1] = (uint8_t)(((7 - i) * a0 + i * a1 + 3) / 7);
    } else {
        for (int i = 1; i <= 4; i++)
            alphas[i + 1] = (uint8_t)(((5 - i) * a0 + i * a1 + 2) / 5);
        alphas[6] = 0;
        alphas[7] = 255;
    }
    // 48-bit alpha index table
    uint64_t aBits = 0;
    for (int i = 2; i < 8; i++)
        aBits |= (uint64_t)block[i] << (8 * (i - 2));

    // Last 8 bytes: color block (same as BC1)
    uint8_t colorOut[4][4][4];
    decompressBC1Block(block + 8, colorOut);

    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            int pixel = row * 4 + col;
            int aIdx = (int)((aBits >> (3 * pixel)) & 0x7);
            out[row][col][0] = colorOut[row][col][0];
            out[row][col][1] = colorOut[row][col][1];
            out[row][col][2] = colorOut[row][col][2];
            out[row][col][3] = alphas[aIdx];
        }
    }
}

bool decompressDDS(const std::string& path,
                   std::vector<unsigned char>& outPixels,
                   int& outWidth, int& outHeight) {
    gli::texture rawTex = gli::load(path);
    if (rawTex.empty() || rawTex.target() != gli::TARGET_2D) return false;

    gli::texture2d tex2D(rawTex);
    if (tex2D.empty()) return false;

    auto extent = tex2D.extent(0);
    outWidth = extent.x;
    outHeight = extent.y;
    if (outWidth == 0 || outHeight == 0) return false;

    gli::format fmt = tex2D.format();
    bool isBC1 = (fmt == gli::FORMAT_RGB_DXT1_UNORM_BLOCK8 ||
                  fmt == gli::FORMAT_RGB_DXT1_SRGB_BLOCK8 ||
                  fmt == gli::FORMAT_RGBA_DXT1_UNORM_BLOCK8 ||
                  fmt == gli::FORMAT_RGBA_DXT1_SRGB_BLOCK8);
    bool isBC3 = (fmt == gli::FORMAT_RGBA_DXT5_UNORM_BLOCK16 ||
                  fmt == gli::FORMAT_RGBA_DXT5_SRGB_BLOCK16);

    if (!isBC1 && !isBC3) return false; // unsupported, let gli try

    size_t blockSize = isBC1 ? 8 : 16;
    int bw = (outWidth + 3) / 4;
    int bh = (outHeight + 3) / 4;

    const uint8_t* src = static_cast<const uint8_t*>(tex2D.data(0, 0, 0));
    outPixels.resize((size_t)outWidth * outHeight * 4);

    for (int by = 0; by < bh; by++) {
        for (int bx = 0; bx < bw; bx++) {
            const uint8_t* block = src + ((size_t)by * bw + bx) * blockSize;
            uint8_t decoded[4][4][4];
            if (isBC1) decompressBC1Block(block, decoded);
            else       decompressBC3Block(block, decoded);

            for (int row = 0; row < 4; row++) {
                int py = by * 4 + row;
                if (py >= outHeight) break;
                for (int col = 0; col < 4; col++) {
                    int px = bx * 4 + col;
                    if (px >= outWidth) break;
                    size_t off = ((size_t)py * outWidth + px) * 4;
                    outPixels[off + 0] = decoded[row][col][0];
                    outPixels[off + 1] = decoded[row][col][1];
                    outPixels[off + 2] = decoded[row][col][2];
                    outPixels[off + 3] = decoded[row][col][3];
                }
            }
        }
    }
    return true;
}