/*******************************************************************************
* Copyright 2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/
#include "cpu/rv64/rvv_gemm_s8s8s32.hpp"
#include "common/dnnl_thread.hpp"
#include "cpu/rv64/rvv_postops.hpp"
#include "cpu/rv64/rvv_gemm_int8_kernel.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

// This routine applies bias and post-ops after the core computation is done.
void rvv_apply_bias_and_postops(int32_t *dst, const dim_t M, const dim_t N,
        const dim_t batch, const int32_t *bias,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {
    parallel_nd(batch, M, [&](dim_t b, dim_t m) {
        int32_t *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e32m4(N - n0);
            vint32m4_t acc = __riscv_vle32_v_i32m4(&dst_base_ptr[n0], vl);

            if (bias) {
                if (bias_d.nelems() == 1) {
                    acc = __riscv_vadd_vx_i32m4(acc, bias[0], vl);
                } else {
                    // Simplified bias handling for this refactoring.
                    // A more robust implementation would handle all broadcast cases.
                    const int32_t *bias_ptr = bias + n0;
                    vint32m4_t bias_vec = __riscv_vle32_v_i32m4(bias_ptr, vl);
                    acc = __riscv_vadd_vv_i32m4(acc, bias_vec, vl);
                }
            }

            acc = postops_handler.apply(acc, vl);
            __riscv_vse32_v_i32m4(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

rvv_gemm_s8s8s32_t::rvv_gemm_s8s8s32_t(const pd_t *apd) : primitive_t(apd) {}

status_t rvv_gemm_s8s8s32_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const int8_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(int32_t *, DNNL_ARG_DST);
    const int32_t *bias = CTX_IN_MEM(const int32_t *, DNNL_ARG_BIAS);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];
    const dim_t M = src_dims[ndims - 2];
    const dim_t N = weights_d.dims()[weights_d.ndims() - 1];

    // Execute core kernel
    if (pd()->is_row_major(src_d)) {
        if (pd()->is_col_major(weights_d)) {
            rvv_gemm_int8_kernel_colmajor<int8_t, int8_t>(
                    src, weights, dst, src_d, weights_d, pd()->src_zero_point_,
                    pd()->weights_zero_point_);
        } else {
            rvv_gemm_int8_kernel_rowmajor<int8_t, int8_t>(
                    src, weights, dst, src_d, weights_d, pd()->src_zero_point_,
                    pd()->weights_zero_point_);
        }
    } else { // src is col-major
        // Note: weights must be row-major in this case, checked in pd_t
        rvv_gemm_int8_kernel_colmajor_src<int8_t, int8_t>(
                src, weights, dst, src_d, weights_d, pd()->src_zero_point_,
                pd()->weights_zero_point_);
    }

    // Apply bias and post-ops
    if (bias || !post_ops.has_default_values()) {
        rvv_apply_bias_and_postops(
                dst, M, N, batch, bias, dst_d, bias_d, postops_handler);
    }

    return status::success;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl
