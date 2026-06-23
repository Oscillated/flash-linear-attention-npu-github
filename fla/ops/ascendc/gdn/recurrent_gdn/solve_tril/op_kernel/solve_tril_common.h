/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#ifndef SOLVE_TRIL_COMMON_H
#define SOLVE_TRIL_COMMON_H

#include "kernel_operator.h"

// ========== Ascend950 平台检测 / UB 优化开关 ==========
// 经实测：Ascend950(arch35) 上把"辅助矩阵在 AIC 核的 UB 上自生成 + UB↔L1 直通"
// 的方案跑不通——AIC(cube)核访问 UB / 执行向量指令(Duplicate/Muls/DataCopy↔UB)
// 得到的是 0/垃圾值（BT=16 输出恒为 0 即为佐证）。因此暂时关闭该 UB 优化路径，
// 在 950 上沿用经验证可用的 910b GM 通路（AIV 生成 I/-I/Zero 到 GM，AIC 经
// GM 加载 + GM scratch 中转）。KERNEL_TYPE_MIX_AIC_1_2 在 950 上同样可用。
//
// 待确认 arch35 的 AIC 核确实可访问 UB 后，再将本宏改回基于
// (__CCE_AICORE__ == 310) 的检测以启用 UB 优化。
#define SOLVE_TRIL_PLATFORM_ASCEND950 0

// ========== MBH 诊断开关（默认关闭）==========
// 打开后，BT>16 的调试路径跳过 RecursiveMerge，仅做 X×I 写回（X=mch_out）。
// 期望输出 == mch_out（块对角逆，非对角块为 0）。用于隔离"多分形 matmul + 写回"
// 是否正确：若输出对角块==mch_out 且非对角==0，则 matmul/store 正常，bug 在
// ExtractBlocksToSlot / L0CToSlot / L0C 累加；否则多分形 matmul 或写回本身有问题。
#ifndef SOLVE_TRIL_MBH_PASSTHROUGH
#define SOLVE_TRIL_MBH_PASSTHROUGH 0
#endif

#if SOLVE_TRIL_PLATFORM_ASCEND950
// Ascend950: 纯 AIC 模式，无 AIC↔AIV 同步需求
// 辅助矩阵在 UB 上生成，无需 GM workspace slot
#else
// AIC/AIV 同步标志常量
constexpr uint64_t SYNC_AIV_AIC_FLAG_SOLVE = 3;
constexpr uint64_t SYNC_AIC_AIV_FLAG_SOLVE = 5;

// GM 共享 workspace slot（ND 行优先）
constexpr int32_t GM_WS_I    = 0;
constexpr int32_t GM_WS_INEG = 1;
constexpr int32_t GM_WS_ZERO = 2;
constexpr int32_t GM_NUM_SHARED_SLOTS = 3;

// AIV 核生成辅助矩阵的参数
constexpr int32_t ROWS_PER_AIV_CORE = 8;
constexpr int32_t DIAG_BLOCK_ELEMS  = ROWS_PER_AIV_CORE * 16;  // 8x16

// 8x16 ND 块中对角 mask（偶数条带）
// 对角在 col 0..7: elem = i*16 + i
constexpr uint64_t DIAG_MASK_8X16_EVEN[2] = {
    0x0008000400020001ULL,
    0x0080004000200010ULL
};
#endif

#endif  // SOLVE_TRIL_COMMON_H
