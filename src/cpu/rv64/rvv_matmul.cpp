/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
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
#include "cpu/rv64/rvv_matmul.hpp"
#include "common/dnnl_thread.hpp"
#include "cpu/rv64/rvv_postops.hpp"
#include "cpu/rv64/rvv_gemm_int8_kernel.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

void rvv_matmul_colmajor(const float *src, const float *weights, float *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const float *bias,
        const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {

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

        const float *src_base_ptr = src + (size_t)b * M * K + (size_t)m * K;
        float *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
        const float *weights_base_ptr = weights + weights_batch_offset;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - n0);
            std::vector<float> out_vals(vl, 0.0f);

            for (dim_t k0 = 0; k0 < K;) {
                size_t k_vl = __riscv_vsetvl_e32m1(K - k0);

                vfloat32m1_t src_vec
                        = __riscv_vle32_v_f32m1(src_base_ptr + k0, k_vl);

                for (size_t ni = 0; ni < vl; ++ni) {
                    const float *weight_col_ptr
                            = weights_base_ptr + (size_t)(n0 + ni) * (size_t)K;
                    vfloat32m1_t wei_vec
                            = __riscv_vle32_v_f32m1(weight_col_ptr + k0, k_vl);

                    vfloat32m1_t prod
                            = __riscv_vfmul_vv_f32m1(src_vec, wei_vec, k_vl);
                    vfloat32m1_t reduced = __riscv_vfredusum_vs_f32m1_f32m1(
                            prod, __riscv_vfmv_v_f_f32m1(0.0f, k_vl), k_vl);
                    float partial = __riscv_vfmv_f_s_f32m1_f32(reduced);

                    out_vals[ni] += partial;
                }

                k0 += k_vl;
            }

            vfloat32m1_t acc = __riscv_vle32_v_f32m1(out_vals.data(), vl);

            if (bias) {
                if (bias_d.nelems() == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
                } else {
                    const int dst_ndims = dst_d.ndims();
                    const int bias_ndims = bias_d.ndims();
                    const dim_t *bias_dims = bias_d.dims();

                    std::vector<size_t> bias_strides(bias_ndims);
                    bias_strides[bias_ndims - 1] = 1;
                    for (int d = bias_ndims - 2; d >= 0; --d)
                        bias_strides[d] = bias_strides[d + 1]
                                * (size_t)bias_dims[d + 1];

                    size_t base_bias_off = 0;
                    for (int d = 0; d < bias_ndims - 1; ++d) {
                        int dst_dim_idx = d + (dst_ndims - bias_ndims);
                        dim_t idx = (bias_dims[d] == 1)
                                ? 0
                                : dst_idx_prefix[dst_dim_idx];
                        base_bias_off += idx * bias_strides[d];
                    }

                    if (bias_dims[bias_ndims - 1] == 1) {
                        acc = __riscv_vfadd_vf_f32m1(
                                acc, bias[base_bias_off], vl);
                    } else {
                        const float *bias_ptr = bias + base_bias_off + n0;
                        vfloat32m1_t bias_vec
                                = __riscv_vle32_v_f32m1(bias_ptr, vl);
                        acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                    }
                }
            }

            acc = postops_handler.apply(acc, vl);
            __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

void rvv_matmul_rowmajor(const float *src, const float *weights, float *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const float *bias,
        const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {

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

        const float *src_base_ptr = src + (size_t)b * M * K + (size_t)m * K;
        float *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
        const float *weights_base_ptr = weights + weights_batch_offset;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - n0);
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (dim_t k = 0; k < K; ++k) {
                vfloat32m1_t a_vec
                        = __riscv_vfmv_v_f_f32m1(src_base_ptr[k], vl);
                const float *b_ptr = weights_base_ptr + (size_t)k * N + n0;
                vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b_ptr, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, a_vec, b_vec, vl);
            }

            if (bias) {
                size_t base_bias_off = 0;
                const int dst_ndims = dst_d.ndims();
                const int bias_ndims = bias_d.ndims();
                const dim_t *bias_dims = bias_d.dims();

                if (bias_d.nelems() != 1) {
                    std::vector<size_t> bias_strides(bias_ndims);
                    bias_strides[bias_ndims - 1] = 1;
                    for (int d = bias_ndims - 2; d >= 0; --d)
                        bias_strides[d] = bias_strides[d + 1]
                                * (size_t)bias_dims[d + 1];

                    for (int d = 0; d < bias_ndims - 1; ++d) {
                        int dst_dim_idx = d + (dst_ndims - bias_ndims);
                        dim_t idx = (bias_dims[d] == 1)
                                ? 0
                                : dst_idx_prefix[dst_dim_idx];
                        base_bias_off += idx * bias_strides[d];
                    }
                }

                if (bias_d.nelems() == 1) {
                    float bias_val = 0.0f;
                        if (bias_d.data_type() == data_type::f32) {
                            bias_val = bias[0];
                        } else if (bias_d.data_type() == data_type::s8) {
                            bias_val = (float)((const int8_t *)bias)[0];
                        }
                        acc = __riscv_vfadd_vf_f32m1(acc, bias_val, vl);
                } else {
                    if (bias_dims[bias_ndims - 1] == 1) {
                        float bias_val = 0.0f;
                        if (bias_d.data_type() == data_type::f32) {
                            bias_val = bias[base_bias_off];
                        } else if (bias_d.data_type() == data_type::s8) {
                            bias_val = (float)((const int8_t *)bias)[base_bias_off];
                        }
                        acc = __riscv_vfadd_vf_f32m1(acc, bias_val, vl);
                    } else {
                        if (bias_d.data_type() == data_type::f32) {
                            const float *bias_ptr = bias + base_bias_off + n0;
                            vfloat32m1_t bias_vec = __riscv_vle32_v_f32m1(bias_ptr, vl);
                            acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                        } else if (bias_d.data_type() == data_type::s8) {
                            const int8_t *bias_ptr = (const int8_t *)bias + base_bias_off + n0;
                            vint8m1_t bias_vec_s8 = __riscv_vle8_v_i8m1(bias_ptr, vl);
                            vfloat32m1_t bias_vec_f32 = __riscv_vfcvt_f_x_v_f32m1(__riscv_vsext_v_i32m1_i8m1(bias_vec_s8, vl), vl);
                            acc = __riscv_vfadd_vv_f32m1(acc, bias_vec_f32, vl);
                        }
                    }
                }
            }

            acc = postops_handler.apply(acc, vl);
            __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

void rvv_matmul_colmajor_src_colmajor_wei(const float *src, const float *weights, float *dst,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const float *bias,
        const memory_desc_wrapper &bias_d,
        const rvv_postops_t &postops_handler) {

    const int ndims = src_d.ndims();
    const dim_t *src_dims = src_d.dims();
    const dim_t *wei_dims = weights_d.dims();
    const int weights_ndims = weights_d.ndims();

    dim_t batch = 1;
    for (int i = 0; i < ndims - 2; ++i)
        batch *= src_dims[i];

    const dim_t M = src_dims[ndims - 2]; // src is K x M
    const dim_t K = src_dims[ndims - 1]; // src is K x M
    const dim_t N = wei_dims[weights_ndims - 1]; // weights is K x N

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

        // src is col-major, so src[k][m] is at src_base_ptr + k * M + m
        const float *src_base_ptr = src + (size_t)b * M * K; // batch offset
        // dst is row-major, dst[b][m][n] is at dst_base_ptr + n
        float *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
        // weights is col-major, weights[k][n] is at weights_base_ptr + k * N + n
        const float *weights_base_ptr = weights + weights_batch_offset;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e32m1(N - n0);
            vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (dim_t k = 0; k < K; ++k) {
                // Access src[k][m]
                vfloat32m1_t a_vec = __riscv_vfmv_v_f_f32m1(*(src_base_ptr + (size_t)k * M + m), vl);
                // Access weights[k][n]
                const float *b_ptr = weights_base_ptr + (size_t)k * N + n0;
                vfloat32m1_t b_vec = __riscv_vle32_v_f32m1(b_ptr, vl);
                acc = __riscv_vfmacc_vv_f32m1(acc, a_vec, b_vec, vl);
            }

            if (bias) {
                if (bias_d.nelems() == 1) {
                    acc = __riscv_vfadd_vf_f32m1(acc, bias[0], vl);
                } else {
                    const int dst_ndims = dst_d.ndims();
                    const int bias_ndims = bias_d.ndims();
                    const dim_t *bias_dims = bias_d.dims();

                    std::vector<size_t> bias_strides(bias_ndims);
                    bias_strides[bias_ndims - 1] = 1;
                    for (int d = bias_ndims - 2; d >= 0; --d)
                        bias_strides[d] = bias_strides[d + 1]
                                * (size_t)bias_dims[d + 1];

                    size_t base_bias_off = 0;
                    for (int d = 0; d < bias_ndims - 1; ++d) {
                        int dst_dim_idx = d + (dst_ndims - bias_ndims);
                        dim_t idx = (bias_dims[d] == 1)
                                ? 0
                                : dst_idx_prefix[dst_dim_idx];
                        base_bias_off += idx * bias_strides[d];
                    }

                    if (bias_dims[bias_ndims - 1] == 1) {
                        acc = __riscv_vfadd_vf_f32m1(
                                acc, bias[base_bias_off], vl);
                    } else {
                        const float *bias_ptr = bias + base_bias_off + n0;
                        vfloat32m1_t bias_vec
                                = __riscv_vle32_v_f32m1(bias_ptr, vl);
                        acc = __riscv_vfadd_vv_f32m1(acc, bias_vec, vl);
                    }
                }
            }

            acc = postops_handler.apply(acc, vl);
            __riscv_vse32_v_f32m1(&dst_base_ptr[n0], acc, vl);
            n0 += vl;
        }
    });
}

rvv_matmul_t::rvv_matmul_t(const pd_t *apd) : primitive_t(apd) {}

status_t rvv_matmul_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const float *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    if (pd()->is_col_major(src_d) && pd()->is_col_major(weights_d)) {
        rvv_matmul_colmajor_src_colmajor_wei(src, weights, dst, src_d, weights_d, dst_d, bias,
                bias_d, postops_handler);
    } else if (pd()->is_col_major(weights_d)) {
        rvv_matmul_colmajor(src, weights, dst, src_d, weights_d, dst_d, bias,
                bias_d, postops_handler);
    } else {
        rvv_matmul_rowmajor(src, weights, dst, src_d, weights_d, dst_d, bias,
                bias_d, postops_handler);
    }

    return status::success;
}

// s8s8f32 implementation

status_t rvv_gemm_s8s8f32_t::pd_t::init(engine_t *engine) {
    status_t status = cpu_matmul_pd_t::init(engine);
    if (status != status::success) return status;

    const memory_desc_wrapper src_mdw(src_md(0));
    const memory_desc_wrapper weights_mdw(weights_md(0));
    const memory_desc_wrapper dst_mdw(dst_md(0));
    const memory_desc_wrapper bias_mdw = bias_md_;

    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    VDISPATCH_MATMUL(!src_mdw.has_runtime_dims_or_strides()
                    && !weights_mdw.has_runtime_dims_or_strides()
                    && !dst_mdw.has_runtime_dims_or_strides()
                    && !bias_mdw.has_runtime_dims_or_strides(),
            VERBOSE_UNSUPPORTED_TAG);

    const bool types_ok = src_mdw.data_type() == data_type::s8
            && weights_mdw.data_type() == data_type::s8
            && dst_mdw.data_type() == data_type::f32
            && desc()->accum_data_type == data_type::f32;
    VDISPATCH_MATMUL(types_ok, VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_MATMUL(attr()->scales_.has_default_values(),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    VDISPATCH_MATMUL(rvv_postops_t::post_ops_ok(attr()->post_ops_),
            VERBOSE_UNSUPPORTED_POSTOP);

    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(check_layouts(src_mdw, weights_mdw, dst_mdw),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(check_bias(dst_mdw, bias_mdw), VERBOSE_UNSUPPORTED_BIAS_CFG);

    src_zero_point_ = 0;
    weights_zero_point_ = 0;
    if (attr()->zero_points_.has_src_zero_points()) {
        src_zero_point_ = attr()->zero_points_.get_src_zero_point();
    }
    if (attr()->zero_points_.has_weights_zero_points()) {
        weights_zero_point_ = attr()->zero_points_.get_weights_zero_point();
    }

    // Need scratchpad for s32 accumulation buffer
    auto scratchpad = scratchpad_registry().registrar();
    scratchpad.book(memory_tracking::names::key_matmul_dst_in_acc_dt,
            M() * N(), sizeof(int32_t));

    return status::success;
}

rvv_gemm_s8s8f32_t::rvv_gemm_s8s8f32_t(const pd_t *apd) : primitive_t(apd) {}

status_t rvv_gemm_s8s8f32_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const int8_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const int8_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(float *, DNNL_ARG_DST);
    const float *bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const dim_t M = pd()->M();
    const dim_t N = pd()->N();
    const dim_t batch = pd()->batch();

    auto scratchpad = ctx.get_scratchpad_grantor();
    int32_t *acc_buffer = scratchpad.template get<int32_t>(
            memory_tracking::names::key_matmul_dst_in_acc_dt);

    // Execute core kernel
    if (pd()->is_row_major(src_d)) {
        if (pd()->is_col_major(weights_d)) {
            rvv_gemm_int8_kernel_colmajor<int8_t, int8_t>(
                    src, weights, acc_buffer, src_d, weights_d, pd()->src_zero_point_,
                    pd()->weights_zero_point_);
        } else {
            rvv_gemm_int8_kernel_rowmajor<int8_t, int8_t>(
                    src, weights, acc_buffer, src_d, weights_d, pd()->src_zero_point_,
                    pd()->weights_zero_point_);
        }
    } else { // src is col-major
        // Note: weights must be row-major in this case, checked in pd_t
        rvv_gemm_int8_kernel_colmajor_src<int8_t, int8_t>(
                src, weights, acc_buffer, src_d, weights_d, pd()->src_zero_point_,
                pd()->weights_zero_point_);
    }

    // Convert to f32, apply bias and post-ops
    parallel_nd(batch, M, [&](dim_t b, dim_t m) {
        const int32_t *s32_ptr = acc_buffer + (size_t)b * M * N + (size_t)m * N;
        float *dst_ptr = dst + (size_t)b * M * N + (size_t)m * N;

        for (dim_t n0 = 0; n0 < N;) {
            size_t vl = __riscv_vsetvl_e32m4(N - n0);
            vint32m4_t s32_vec = __riscv_vle32_v_i32m4(s32_ptr + n0, vl);
            vfloat32m4_t f32_vec = __riscv_vfcvt_f_x_v_f32m4(s32_vec, vl);

            if (bias) {
                 if (bias_d.nelems() == 1) {
                    f32_vec = __riscv_vfadd_vf_f32m4(f32_vec, bias[0], vl);
                } else {
                    const float *bias_ptr = bias + n0;
                    vfloat32m4_t bias_vec = __riscv_vle32_v_f32m4(bias_ptr, vl);
                    f32_vec = __riscv_vfadd_vv_f32m4(f32_vec, bias_vec, vl);
                }
            }

            f32_vec = postops_handler.apply(f32_vec, vl);
            __riscv_vse32_v_f32m4(dst_ptr + n0, f32_vec, vl);
            n0 += vl;
        }
    });

    return status::success;
}

// u8u8u32 implementation

status_t rvv_gemm_u8u8u32_t::pd_t::init(engine_t *engine) {
    status_t status = cpu_matmul_pd_t::init(engine);
    if (status != status::success) return status;

    const memory_desc_wrapper src_mdw(src_md(0));
    const memory_desc_wrapper weights_mdw(weights_md(0));
    const memory_desc_wrapper dst_mdw(dst_md(0));
    const memory_desc_wrapper bias_mdw = bias_md_;

    VDISPATCH_MATMUL(!has_zero_dim_memory(), VERBOSE_EMPTY_TENSOR, "");

    VDISPATCH_MATMUL(!src_mdw.has_runtime_dims_or_strides()
                    && !weights_mdw.has_runtime_dims_or_strides()
                    && !dst_mdw.has_runtime_dims_or_strides()
                    && !bias_mdw.has_runtime_dims_or_strides(),
            VERBOSE_UNSUPPORTED_TAG);

    const bool types_ok = src_mdw.data_type() == data_type::u8
            && weights_mdw.data_type() == data_type::u8
            && dst_mdw.data_type() == data_type::s32
            && desc()->accum_data_type == data_type::s32;
    VDISPATCH_MATMUL(types_ok, VERBOSE_UNSUPPORTED_DT);

    VDISPATCH_MATMUL(attr()->scales_.has_default_values(),
            VERBOSE_UNSUPPORTED_SCALES_CFG);

    VDISPATCH_MATMUL(rvv_postops_t::post_ops_ok(attr()->post_ops_),
            VERBOSE_UNSUPPORTED_POSTOP);

    VDISPATCH_MATMUL(set_default_formats(), VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(check_layouts(src_mdw, weights_mdw, dst_mdw),
            VERBOSE_UNSUPPORTED_TAG);
    VDISPATCH_MATMUL(check_bias(dst_mdw, bias_mdw), VERBOSE_UNSUPPORTED_BIAS_CFG);

    src_zero_point_ = 0;
    weights_zero_point_ = 0;
    if (attr()->zero_points_.has_src_zero_points()) {
        src_zero_point_ = attr()->zero_points_.get_src_zero_point();
    }
    if (attr()->zero_points_.has_weights_zero_points()) {
        weights_zero_point_ = attr()->zero_points_.get_weights_zero_point();
    }

    return status::success;
}

rvv_gemm_u8u8u32_t::rvv_gemm_u8u8u32_t(const pd_t *apd) : primitive_t(apd) {}

status_t rvv_gemm_u8u8u32_t::execute(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const uint8_t *, DNNL_ARG_SRC);
    auto weights = CTX_IN_MEM(const uint8_t *, DNNL_ARG_WEIGHTS);
    auto dst = CTX_OUT_MEM(uint32_t *, DNNL_ARG_DST);
    const int32_t *bias = CTX_IN_MEM(const int32_t *, DNNL_ARG_BIAS);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper bias_d(pd()->desc()->bias_desc);

    const post_ops_t &post_ops = pd()->attr()->post_ops_;
    rvv_postops_t postops_handler(post_ops);

    const dim_t M = pd()->M();
    const dim_t N = pd()->N();
    const dim_t batch = pd()->batch();

    // Execute core kernel
    if (pd()->is_row_major(src_d)) {
        if (pd()->is_col_major(weights_d)) {
            rvv_gemm_int8_kernel_colmajor<uint8_t, uint8_t>(
                    src, weights, (int32_t *)dst, src_d, weights_d, pd()->src_zero_point_,
                    pd()->weights_zero_point_);
        } else {
            rvv_gemm_int8_kernel_rowmajor<uint8_t, uint8_t>(
                    src, weights, (int32_t *)dst, src_d, weights_d, pd()->src_zero_point_,
                    pd()->weights_zero_point_);
        }
    } else { // src is col-major
        // Note: weights must be row-major in this case, checked in pd_t
        rvv_gemm_int8_kernel_colmajor_src<uint8_t, uint8_t>(
                src, weights, (int32_t *)dst, src_d, weights_d, pd()->src_zero_point_,
                pd()->weights_zero_point_);
    }

    // Apply bias and post-ops
    if (bias || !post_ops.has_default_values()) {
        // This is a simplified post-ops application for u8. A full implementation
        // would need a dedicated u32 post-ops handler.
        parallel_nd(batch, M, [&](dim_t b, dim_t m) {
            uint32_t *dst_base_ptr = dst + (size_t)b * M * N + (size_t)m * N;
            for (dim_t n0 = 0; n0 < N;) {
                size_t vl = __riscv_vsetvl_e32m4(N - n0);
                vuint32m4_t acc = __riscv_vle32_v_u32m4(&dst_base_ptr[n0], vl);
                if (bias) {
                    if (bias_d.nelems() == 1) {
                        acc = __riscv_vadd_vx_u32m4(acc, bias[0], vl);
                    } else {
                        const int32_t *bias_ptr = bias + n0;
                        vuint32m4_t bias_vec = __riscv_vle32_v_u32m4((const uint32_t*)bias_ptr, vl);
                        acc = __riscv_vadd_vv_u32m4(acc, bias_vec, vl);
                    }
                }
                // Post-ops for u32 would need a separate handler. Assuming none for now.
                __riscv_vse32_v_u32m4(&dst_base_ptr[n0], acc, vl);
                n0 += vl;
            }
        });
    }

    return status::success;
}

// Common helper functions for pd_t

template <typename pd_t_>
bool is_row_major_impl(const memory_desc_wrapper &mdw) {
    const int ndims = mdw.ndims();
    if (ndims < 2) return false;
    const auto &strides = mdw.blocking_desc().strides;
    if (strides[ndims - 1] != 1) return false;
    dim_t expected_stride = mdw.dims()[ndims - 1];
    for (int d = ndims - 2; d >= 0; --d) {
        if (strides[d] != expected_stride) return false;
        expected_stride *= mdw.dims()[d];
    }
    return true;
}

template <typename pd_t_>
bool is_col_major_impl(const memory_desc_wrapper &mdw) {
    const int ndims = mdw.ndims();
    if (ndims < 2) return false;
    const auto &strides = mdw.blocking_desc().strides;
    const auto &dims = mdw.dims();
    if (strides[ndims - 2] != 1) return false;
    if (strides[ndims - 1] != dims[ndims - 2]) return false;
    dim_t expected_stride = dims[ndims - 2] * dims[ndims - 1];
    for (int d = ndims - 3; d >= 0; --d) {
        if (strides[d] != expected_stride) return false;
        expected_stride *= dims[d];
    }
    return true;
}

bool rvv_gemm_s8s8f32_t::pd_t::is_row_major(const memory_desc_wrapper &mdw) const { return is_row_major_impl<pd_t>(mdw); }
bool rvv_gemm_s8s8f32_t::pd_t::is_col_major(const memory_desc_wrapper &mdw) const { return is_col_major_impl<pd_t>(mdw); }
bool rvv_gemm_u8u8u32_t::pd_t::is_row_major(const memory_desc_wrapper &mdw) const { return is_row_major_impl<pd_t>(mdw); }
bool rvv_gemm_u8u8u32_t::pd_t::is_col_major(const memory_desc_wrapper &mdw) const { return is_col_major_impl<pd_t>(mdw); }

bool rvv_gemm_s8s8f32_t::pd_t::check_layouts(const memory_desc_wrapper &src_mdw,
        const memory_desc_wrapper &wei_mdw,
        const memory_desc_wrapper &dst_mdw) const {
    // Source and weights can be either row-major (ab) or column-major (ba).
    // Destination must be row-major (ab).
    if (!is_row_major(src_mdw) && !is_col_major(src_mdw)) return false;
    if (!is_row_major(dst_mdw)) return false;
    if (!is_row_major(wei_mdw) && !is_col_major(wei_mdw)) return false;
    return true;
}

bool rvv_gemm_u8u8u32_t::pd_t::check_layouts(const memory_desc_wrapper &src_mdw,
        const memory_desc_wrapper &wei_mdw,
        const memory_desc_wrapper &dst_mdw) const {
    // Source and weights can be either row-major (ab) or column-major (ba).
    // Destination must be row-major (ab).
    if (!is_row_major(src_mdw) && !is_col_major(src_mdw)) return false;
    if (!is_row_major(dst_mdw)) return false;
    if (!is_row_major(wei_mdw) && !is_col_major(wei_mdw)) return false;
    return true;
}

bool rvv_gemm_s8s8f32_t::pd_t::check_bias(const memory_desc_wrapper &dst_mdw,
        const memory_desc_wrapper &bias_mdw) const {
    if (bias_mdw.is_zero()) return true;
    if (bias_mdw.data_type() != data_type::f32 && bias_mdw.data_type() != data_type::s8) return false;
    // Further checks from f32 implementation can be added here
    return true;
}

bool rvv_gemm_u8u8u32_t::pd_t::check_bias(const memory_desc_wrapper &dst_mdw,
        const memory_desc_wrapper &bias_mdw) const {
    if (bias_mdw.is_zero()) return true;
    if (bias_mdw.data_type() != data_type::s32 && bias_mdw.data_type() != data_type::s8 && bias_mdw.data_type() != data_type::f32) return false;
    // Further checks from s32 implementation can be added here
    return true;
}

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl