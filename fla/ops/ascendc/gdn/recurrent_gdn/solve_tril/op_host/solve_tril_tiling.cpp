/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

/*!
 * \file solve_tril_tiling.cpp
 * \brief Tiling for SolveTril: input [B,S,H,BT] or [T,H,BT], BT in {16,32,64,128}
 */

#include "solve_tril_tiling.h"
#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "../op_kernel/solve_tril_common.h"

namespace optiling {

using Ops::Base::CeilDiv;

constexpr size_t WORKSPACE_NUM = 1;
constexpr uint32_t WS_SYS_SIZE = 0U;

static ge::graphStatus GetPlatformInfo(gert::TilingContext* context, int64_t* coreNum)
{
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    *coreNum = ascendcPlatform.GetCoreNumAiv();
    OP_CHECK_IF(*coreNum == 0, OP_LOGE(context, "coreNum is 0"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SolveTrilTilingFunc(gert::TilingContext* context)
{
    // 1. Platform info
    int64_t coreNum;
    OP_CHECK_IF(GetPlatformInfo(context, &coreNum) != ge::GRAPH_SUCCESS,
                OP_LOGE(context, "GetPlatformInfo error"), return ge::GRAPH_FAILED);

    // 2. Parse input shape
    auto inputShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, inputShape);
    auto shape = inputShape->GetStorageShape();
    int64_t dimNum = shape.GetDimNum();

    int64_t B, S, H, BT;
    if (dimNum == 4) {
        // BSND: [B, S, H, BT]
        B  = shape.GetDim(0);
        S  = shape.GetDim(1);
        H  = shape.GetDim(2);
        BT = shape.GetDim(3);
    } else if (dimNum == 3) {
        // TND: [T, H, BT]
        B  = 1;
        S  = shape.GetDim(0);
        H  = shape.GetDim(1);
        BT = shape.GetDim(2);
    } else {
        OP_LOGE(context, "SolveTril: unsupported rank %ld, expected 3 or 4", dimNum);
        return ge::GRAPH_FAILED;
    }

    // 3. Validate BT
    if (BT != 16 && BT != 32 && BT != 64 && BT != 128) {
        OP_LOGE(context, "SolveTril: BT=%ld not in {16,32,64,128}", BT);
        return ge::GRAPH_FAILED;
    }

    // 4. Derived values
    int64_t NT = CeilDiv(S, BT);
    int64_t numLeafBlocks = BT / LEAF_BLOCK_SIZE;
    int64_t mbhLevels = 0;
    if (BT == 32)      mbhLevels = 1;
    else if (BT == 64) mbhLevels = 2;
    else if (BT == 128)mbhLevels = 3;

    // 5. Detect varlen
    int64_t isVarlen = 0;
    auto cuSeqlensShape = context->GetOptionalInputShape(1);
    if (cuSeqlensShape != nullptr) isVarlen = 1;

    // 6. Total tasks
    int64_t totalTasks;
    if (isVarlen) {
        auto chunkIndicesShape = context->GetOptionalInputShape(2);
        if (chunkIndicesShape == nullptr) {
            OP_LOGE(context, "SolveTril: varlen requires chunk_indices_out");
            return ge::GRAPH_FAILED;
        }
        totalTasks = chunkIndicesShape->GetStorageShape().GetDim(0);
    } else {
        totalTasks = B * H * NT;
    }

    // 7. Multi-core split
    int64_t usedCoreNum = totalTasks < coreNum ? totalTasks : coreNum;
    if (usedCoreNum == 0) usedCoreNum = 1;
    int64_t taskPerCore = CeilDiv(totalTasks, usedCoreNum);

    // 8. Workspace
    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    size_t wsSize = WS_SYS_SIZE;
    if (mbhLevels >= 2) {
        wsSize = static_cast<size_t>(totalTasks) * static_cast<size_t>(BT) * BT * sizeof(float);
    }
    currentWorkspace[0] = wsSize;

    // 9. Fill TilingData
    SolveTrilTilingData* tiling = context->GetTilingData<SolveTrilTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    tiling->n = BT;
    tiling->totalTasks = totalTasks;
    tiling->H = H;
    tiling->NT = NT;
    tiling->rowStride = H * BT;
    tiling->mbhLevels = mbhLevels;
    tiling->blockDim = usedCoreNum;
    tiling->taskPerCore = taskPerCore;

    context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

    // 10. TilingKey
    auto inputDtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dType = static_cast<uint32_t>(inputDtype);
    ASCENDC_TPL_SEL_PARAM(context, dType, static_cast<uint32_t>(mbhLevels));

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForSolveTril([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct SolveTrilCompileInfo {};

IMPL_OP_OPTILING(SolveTril)
    .Tiling(SolveTrilTilingFunc)
    .TilingParse<SolveTrilCompileInfo>(TilingParseForSolveTril);

} // namespace optiling
