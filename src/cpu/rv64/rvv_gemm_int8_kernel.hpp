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
#ifndef CPU_RV64_RVV_GEMM_INT8_KERNEL_HPP
#define CPU_RV64_RVV_GEMM_INT8_KERNEL_HPP

#include "common/dnnl_thread.hpp"
#include "common/memory_desc_wrapper.hpp"
#include <riscv_vector.h>
#include <type_traits>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

// Kernel for SRC row-major, WEIGHTS col-major
template <typename T_src, typename T_wgt>
void rvv_gemm_int8_kernel_colmajor(const T_src *src, const T_wgt *weights,
        int32_t *dst, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, int32_t src_zero_point,
        int32_t weights_zero_point) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2];
    const dim_t K = src_dims[ndims - 1];
    const dim_t N = wei_dims[weights_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < weights_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    const bool weights_are_broadcasted = (weights_batch_size == 1 && batch > 1);

    parallel_nd(batch, M, [&](dim_t b, dim_t m) {
        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2) {
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);
        }
        dst_idx_prefix[ndims - 2] = m;

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        }

        const T_src *src_base_ptr = src + (size_t)b * M * K + (size_t)m * K;
        int32_t *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
        const T_wgt *weights_base_ptr = weights + weights_batch_offset;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e8m1(N - n0);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

            for (dim_t k = 0; k < K; ++k) {
                int16_t a_val = (int16_t)src_base_ptr[k] - src_zero_point;
                const T_wgt *b_ptr = weights_base_ptr + (size_t)n0 * K + k;

                vint8m1_t b_vec_s8 = __riscv_vlse8_v_i8m1((const int8_t *)b_ptr, K, vl);

                if (std::is_same<T_wgt, int8_t>::value) {
                    vint16m2_t b_vec_s16 = __riscv_vsext_vf2_i16m2(b_vec_s8, vl);
                    vint16m2_t b_vec_s16_zp = __riscv_vmv_v_x_i16m2(weights_zero_point, vl);
                    b_vec_s16 = __riscv_vsub_vv_i16m2(b_vec_s16, b_vec_s16_zp, vl);
                    acc = __riscv_vwmacc_vx_i32m4(acc, a_val, b_vec_s16, vl);
                } else { // uint8_t
                    vuint8m1_t b_vec_u8 = __riscv_vreinterpret_v_i8m1_u8m1(b_vec_s8);
                    vuint16m2_t b_vec_u16 = __riscv_vzext_vf2_u16m2(b_vec_u8, vl);
                    vuint16m2_t b_vec_u16_zp = __riscv_vmv_v_x_u16m2(weights_zero_point, vl);
                    b_vec_u16 = __riscv_vsub_vv_u16m2(b_vec_u16, b_vec_u16_zp, vl);
                    acc = __riscv_vwmaccsu_vx_i32m4(acc, a_val, b_vec_u16, vl);
                }
            }
            __riscv_vse32_v_i32m4(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

// Kernel for SRC row-major, WEIGHTS row-major
template <typename T_src, typename T_wgt>
void rvv_gemm_int8_kernel_rowmajor(const T_src *src, const T_wgt *weights,
        int32_t *dst, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, int32_t src_zero_point,
        int32_t weights_zero_point) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2];
    const dim_t K = src_dims[ndims - 1];
    const dim_t N = wei_dims[weights_ndims - 1];

    dim_t weights_batch_size = 1;
    for (int i = 0; i < weights_ndims - 2; ++i)
        weights_batch_size *= wei_dims[i];
    const bool weights_are_broadcasted = (weights_batch_size == 1 && batch > 1);

    parallel_nd(batch, M, [&](dim_t b, dim_t m) {
        std::vector<dim_t> dst_idx_prefix(ndims - 1);
        if (ndims > 2) {
            utils::l_dims_by_l_offset(
                    dst_idx_prefix.data(), b, src_dims, ndims - 2);
        }
        dst_idx_prefix[ndims - 2] = m;

        size_t weights_batch_offset = 0;
        if (!weights_are_broadcasted) {
            for (int i = 0; i < weights_ndims - 2; ++i) {
                if (wei_dims[i] != 1) {
                    dim_t b_idx = dst_idx_prefix[i + (ndims - weights_ndims)];
                    weights_batch_offset
                            += b_idx * weights_d.blocking_desc().strides[i];
                }
            }
        }

        const T_src *src_base_ptr = src + (size_t)b * M * K + (size_t)m * K;
        int32_t *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
        const T_wgt *weights_base_ptr = weights + weights_batch_offset;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e8m1(N - n0);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

            for (dim_t k = 0; k < K; ++k) {
                int16_t a_val = (int16_t)src_base_ptr[k] - src_zero_point;
                const T_wgt *b_ptr = weights_base_ptr + (size_t)k * N + n0;

                vint8m1_t b_vec_s8 = __riscv_vle8_v_i8m1((const int8_t *)b_ptr, vl);

                if (std::is_same<T_wgt, int8_t>::value) {
                    vint16m2_t b_vec_s16 = __riscv_vsext_vf2_i16m2(b_vec_s8, vl);
                    vint16m2_t b_vec_s16_zp = __riscv_vmv_v_x_i16m2(weights_zero_point, vl);
                    b_vec_s16 = __riscv_vsub_vv_i16m2(b_vec_s16, b_vec_s16_zp, vl);
                    acc = __riscv_vwmacc_vx_i32m4(acc, a_val, b_vec_s16, vl);
                } else { // uint8_t
                    vuint8m1_t b_vec_u8 = __riscv_vreinterpret_v_i8m1_u8m1(b_vec_s8);
                    vuint16m2_t b_vec_u16 = __riscv_vzext_vf2_u16m2(b_vec_u8, vl);
                    vuint16m2_t b_vec_u16_zp = __riscv_vmv_v_x_u16m2(weights_zero_point, vl);
                    b_vec_u16 = __riscv_vsub_vv_u16m2(b_vec_u16, b_vec_u16_zp, vl);
                    acc = __riscv_vwmaccsu_vx_i32m4(acc, a_val, b_vec_u16, vl);
                }
            }
            __riscv_vse32_v_i32m4(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

// Kernel for SRC column-major, WEIGHTS row-major
template <typename T_src, typename T_wgt>
void rvv_gemm_int8_kernel_colmajor_src(const T_src *src, const T_wgt *weights,
        int32_t *dst, const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &weights_d, int32_t src_zero_point,
        int32_t weights_zero_point) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2];
    const dim_t K = src_dims[ndims - 1];
    const dim_t N = wei_dims[weights_ndims - 1];

    parallel_nd(batch, M, [&](dim_t b, dim_t m) {
        const T_src *src_batch_ptr = src + (size_t)b * M * K;
        const T_wgt *weights_batch_ptr = weights + (size_t)b * K * N;
        int32_t *dst_row_ptr = dst + (size_t)b * M * N + (size_t)m * N;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e8m1(N - n0);
            vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);

            for (dim_t k = 0; k < K; ++k) {
                int16_t a_val = (int16_t)src_batch_ptr[k * M + m] - src_zero_point;
                const T_wgt *b_ptr = weights_batch_ptr + (size_t)k * N + n0;

                vint8m1_t b_vec_s8 = __riscv_vle8_v_i8m1((const int8_t *)b_ptr, vl);

                if (std::is_same<T_wgt, int8_t>::value) {
                    vint16m2_t b_vec_s16 = __riscv_vsext_vf2_i16m2(b_vec_s8, vl);
                    vint16m2_t b_vec_s16_zp = __riscv_vmv_v_x_i16m2(weights_zero_point, vl);
                    b_vec_s16 = __riscv_vsub_vv_i16m2(b_vec_s16, b_vec_s16_zp, vl);
                    acc = __riscv_vwmacc_vx_i32m4(acc, a_val, b_vec_s16, vl);
                } else { // uint8_t
                    vuint8m1_t b_vec_u8 = __riscv_vreinterpret_v_i8m1_u8m1(b_vec_s8);
                    vuint16m2_t b_vec_u16 = __riscv_vzext_vf2_u16m2(b_vec_u8, vl);
                    vuint16m2_t b_vec_u16_zp = __riscv_vmv_v_x_u16m2(weights_zero_point, vl);
                    b_vec_u16 = __riscv_vsub_vv_u16m2(b_vec_u16, b_vec_u16_zp, vl);
                    acc = __riscv_vwmaccsu_vx_i32m4(acc, a_val, b_vec_u16, vl);
                }
            }
            __riscv_vse32_v_i32m4(&dst_row_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_GEMM_INT8_KERNEL_HPP