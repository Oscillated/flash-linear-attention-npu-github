# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Tianjin University, Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/solve_tril_def.cpp
    )
endif()

add_modules_sources(OPTYPE solve_tril ACLNNTYPE aclnn_exclude)

# MBH 调试模式：屏蔽 MCH，MCH 输出/全 0 矩阵/单位矩阵 作为接口入参，单独调试 MBH 模块。
# 正式构建时移除 -DSOLVE_TRIL_MBH_DEBUG_ONLY=1。
add_ops_compile_options(OP_NAME SolveTril OPTIONS
    --cce-auto-sync=off
    -DSOLVE_TRIL_MBH_DEBUG_ONLY=1
)
