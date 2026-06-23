/**
 * Copyright (c) 2026 Tianjin University, Ltd.
 * Licensed under the BSD 3-Clause License.
 */

#include "aclnn_solve_tril.h"
#include <new>

#include "aclnn_kernels/contiguous.h"
#include "acl/acl.h"
#include "aclnn/aclnn_base.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/op_dfx.h"
#include "opdev/make_op_executor.h"
#include "opdev/tensor_view_utils.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct SolveTrilParams {
    const aclTensor *x = nullptr;
    const aclIntArray *cuSeqlens = nullptr;
    const aclIntArray *chunkIndices = nullptr;
    const aclTensor *mchOut = nullptr;
    const aclTensor *zeroMat = nullptr;
    const aclTensor *eyeMat = nullptr;
    int64_t chunkSize = 64;
    const char *layout = "bsnd";
    const aclTensor *xOut = nullptr;
};

static aclnnStatus CheckNotNull(SolveTrilParams params)
{
    CHECK_COND(params.x != nullptr, ACLNN_ERR_PARAM_NULLPTR, "x must not be nullptr.");
    CHECK_COND(params.xOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "xOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus CheckParams(SolveTrilParams params)
{
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(SolveTrilParams &params, aclOpExecutor *executorPtr)
{
    CHECK_COND(DataContiguous(params.x, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "Contiguous x failed.");
    // MBH 调试入参（可选）也需保证连续
    if (params.mchOut != nullptr) {
        CHECK_COND(DataContiguous(params.mchOut, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous mch_out failed.");
    }
    if (params.zeroMat != nullptr) {
        CHECK_COND(DataContiguous(params.zeroMat, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous zero_mat failed.");
    }
    if (params.eyeMat != nullptr) {
        CHECK_COND(DataContiguous(params.eyeMat, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
                   "Contiguous eye_mat failed.");
    }
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSolveTrilGetWorkspaceSize(
    const aclTensor *x,
    const aclIntArray *cuSeqlens,
    const aclIntArray *chunkIndices,
    const aclTensor *mchOut,
    const aclTensor *zeroMat,
    const aclTensor *eyeMat,
    int64_t chunkSize,
    const char *layout,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    SolveTrilParams params{x, cuSeqlens, chunkIndices, mchOut, zeroMat, eyeMat, chunkSize, layout, xOut};

    L2_DFX_PHASE_1(aclnnSolveTril, DFX_IN(x, cuSeqlens, chunkIndices, mchOut, zeroMat, eyeMat), DFX_OUT(xOut));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();

    auto ret = CheckParams(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    CHECK_COND(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID,
               "ParamsDataContiguous failed.");

    auto result = l0op::SolveTril(params.x, params.cuSeqlens, params.chunkIndices,
                                  params.mchOut, params.zeroMat, params.eyeMat,
                                  params.chunkSize, params.layout, params.xOut, executorPtr);
    CHECK_RET(result != nullptr, ACLNN_ERR_PARAM_NULLPTR);

    auto viewCopyResult = l0op::ViewCopy(result, params.xOut, executorPtr);
    CHECK_RET(viewCopyResult != nullptr, ACLNN_ERR_INNER_NULLPTR);

    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnSolveTril(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor, aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnSolveTril);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "SolveTril launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
