#ifndef __SOLVE_TRIL_TILING_DATA_H__
#define __SOLVE_TRIL_TILING_DATA_H__

#include <cstdint>

struct SolveTrilTilingData {
    int64_t n;                // BT (chunk size: 16/32/64/128)
    int64_t totalTasks;       // B * H * NT
    int64_t H;                // num heads
    int64_t NT;               // num chunks per (batch, head) = ceil(S / BT)
    int64_t rowStride;        // H * BT (GM stride between rows in a chunk)
    int64_t mbhLevels;        // 0/1/2/3
    int64_t blockDim;         // num AI cores used
    int64_t taskPerCore;      // tasks assigned per core
};

#endif // __SOLVE_TRIL_TILING_DATA_H__
