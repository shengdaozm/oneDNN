#include "cpu/rv64/matmul/rvv_matmul.hpp"
#include <cstdio>
#include <iostream>
#include "common/c_types_map.hpp"
#include "common/memory_desc.hpp"
#include "common/utils.hpp"
#include "common/dnnl_thread.hpp"
#include "common/math_utils.hpp"
#include "cpu/ref_io_helper.hpp"
#include "cpu/matmul/matmul_utils.hpp"
#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

// Helper function to load value based on data type
template <typename T>
inline float load_value(const void *ptr, size_t offset) {
    const T *typed_ptr = reinterpret_cast<const T *>(ptr);
    return static_cast<float>(typed_ptr[offset]);
}

// Specialization for float
template <>
inline float load_value<float>(const void *ptr, size_t offset) {
    const float *typed_ptr = reinterpret_cast<const float *>(ptr);
    return typed_ptr[offset];
}

// Specialization for f16
template <>
inline float load_value<float16_t>(const void *ptr, size_t offset) {
    const float16_t *typed_ptr = reinterpret_cast<const float16_t *>(ptr);
    return (float)typed_ptr[offset];
}

template <>
status_t riscv_matmul_t<data_type::f32>::execute(const exec_ctx_t &ctx) const {
    // get input&output data
    status_t status = status::success;
    const auto src = CTX_IN_MEM(const float *, DNNL_ARG_SRC);
    const auto weights = CTX_IN_MEM(const float *, DNNL_ARG_WEIGHTS);
    const auto bias = CTX_IN_MEM(const float *, DNNL_ARG_BIAS);
    auto dst = CTX_OUT_CLEAN_MEM(float *, DNNL_ARG_DST, status);
    CHECK(status);

    // get memory_desc_wrapper
    const auto src_d = ctx.memory_mdw(DNNL_ARG_SRC, pd()->src_md());
    const auto weights_d = ctx.memory_mdw(DNNL_ARG_WEIGHTS, pd()->weights_md());
    const auto dst_d = ctx.memory_mdw(DNNL_ARG_DST, pd()->dst_md());
    const auto bia_d = ctx.memory_mdw(DNNL_ARG_BIAS, pd()->weights_md(1));

    // check zero dimensions
    if (src_d.has_zero_dim() || weights_d.has_zero_dim() || dst_d.has_zero_dim()) {
        printf("==Zero dimension detected in src, weights or dst.==\n");
        return status::success;
    }

    // get matmul dimensions
    dnnl::impl::cpu::matmul::matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();
    const dim_t src_stride = helper.get_a_stride(src_d.ndims() - 2);
    const dim_t weights_stride = helper.get_b_stride(weights_d.ndims() - 1);

    // Get attribute related data
    const auto &attr = *pd()->attr();
    const bool has_post_ops = !attr.post_ops_.is_empty();
    const bool with_scales = !attr.scales_.has_default_values();
    
    // Initialize scales
    DEFINE_ARG_SCALES_BUFFER(src_scales, DNNL_ARG_SRC);
    DEFINE_ARG_SCALES_BUFFER(wei_scales, DNNL_ARG_WEIGHTS);
    DEFINE_ARG_SCALES_BUFFER(dst_scales, DNNL_ARG_DST);

    // get mask information for broadcasting
    const int src_mask = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
    const int wei_mask = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
    const int bia_mask = utils::get_dims_mask(dst_d.dims(), bia_d.dims(), ndims);

    // Prepare post ops
    ref_post_ops_t ref_post_ops;
    ref_post_ops.prepare(ctx, attr, dst_d.data_type(), dst_d.dims(), ndims);
    
    // rvv_kernel
    auto rvv_matmul_kernel = [&](const dims_t &dst_dims_idx, dim_t m, dim_t n) -> float {
        dims_t src_dims_idx, weights_dims_idx;
        utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
        utils::copy_dims_with_mask(weights_dims_idx, dst_dims_idx, ndims, wei_mask);
        
        src_dims_idx[ndims - 2] = m;
        weights_dims_idx[ndims - 1] = n;
        auto &src_k_dim = src_dims_idx[ndims - 1];
        auto &wei_k_dim = weights_dims_idx[ndims - 2];

        float result = 0.0f;
        
        // 向量化主循环
        for (dim_t k = 0; k < K; ) {
            size_t vl = __riscv_vsetvl_e32m1(K - k);
            
            // 创建临时数组用于存储数据
            float src_vals[vl], weights_vals[vl];
            
            // 填充临时数组
            for (size_t i = 0; i < vl; ++i) {
                src_k_dim = k + i;
                wei_k_dim = k + i;
                
                const auto src_off = src_d.off_v(src_dims_idx);
                const auto weights_off = weights_d.off_v(weights_dims_idx);
                
                src_vals[i] = load_value<float>(src, src_off);
                if (with_scales && !attr.scales_.has_default_values(DNNL_ARG_SRC)) {
                    src_vals[i] = apply_scale(src_vals[i], src_scales[0]);
                }
                weights_vals[i] = load_value<float>(weights, weights_off);
                if (with_scales && !attr.scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
                    weights_vals[i] = apply_scale(weights_vals[i], wei_scales[0]);
                }
            }
            
            vfloat32m1_t vec_src = __riscv_vle32_v_f32m1(src_vals, vl);
            vfloat32m1_t vec_weights = __riscv_vle32_v_f32m1(weights_vals, vl);
            vfloat32m1_t vec_mul = __riscv_vfmul_vv_f32m1(vec_src, vec_weights, vl);
            vfloat32m1_t vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_mul, __riscv_vfmv_s_f_f32m1(0.0f, 1), vl);
            result += __riscv_vfmv_f_s_f32m1_f32(vec_sum);
            
            k += vl;
        }
        return result;
    };

    // 标量内核（用于小K值或fallback）
    auto scalar_kernel = [&](const dims_t &dst_dims_idx, dim_t m, dim_t n) -> float {
        float acc = 0.0f;
        dims_t src_dims_idx, weights_dims_idx;
        utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
        utils::copy_dims_with_mask(weights_dims_idx, dst_dims_idx, ndims, wei_mask);
        
        src_dims_idx[ndims - 2] = m;
        weights_dims_idx[ndims - 1] = n;
        auto &src_k_dim = src_dims_idx[ndims - 1];
        auto &wei_k_dim = weights_dims_idx[ndims - 2];
        
        for (dim_t k = 0; k < K; ++k) {
            src_k_dim = k;
            wei_k_dim = k;
            const auto src_off = src_d.off_v(src_dims_idx);
            const auto weights_off = weights_d.off_v(weights_dims_idx);
            float s = load_value<float>(src, src_off);
            float w = load_value<float>(weights, weights_off);
            if (with_scales && !attr.scales_.has_default_values(DNNL_ARG_SRC)) {
                s = apply_scale(s, src_scales[0]);
            }
            if (with_scales && !attr.scales_.has_default_values(DNNL_ARG_WEIGHTS)) {
                w = apply_scale(w, wei_scales[0]);
            }
            acc += s * w;
        }
        return acc;
    };

    // bias处理函数
    auto get_bias = [&](const dims_t &dst_dims_idx) -> float {
        if (!bias) return 0.0f;
        dims_t bia_dims_idx;
        utils::copy_dims_with_mask(bia_dims_idx, dst_dims_idx, ndims, bia_mask);
        const auto bias_off = bia_d.off_v(bia_dims_idx);
        return load_value<float>(bias, bias_off);
    };

    // 主计算循环 - 使用OpenMP并行化
    parallel_nd(batch, M, N, [&](dim_t mb, dim_t m, dim_t n) {
        dims_t dst_dims_idx;
        const size_t l_offset = mb * M * N + m * N + n;
        utils::l_dims_by_l_offset(dst_dims_idx, l_offset, dst_d.dims(), ndims);
        
        // 选择使用RVV优化内核或标量内核
        float result = (K>=8) ? rvv_matmul_kernel(dst_dims_idx, m, n)
                              : scalar_kernel(dst_dims_idx, m, n);
        
        // add bias
        result += get_bias(dst_dims_idx);
        
        if (with_scales && !attr.scales_.has_default_values(DNNL_ARG_DST)) {
            result = apply_scale(result, dst_scales[0]);
        }

        if (has_post_ops) {
            ref_post_ops_args_t args;
            args.dst_val = &result;
            args.ctx = &ctx;
            args.ndims = ndims;
            args.dims = dst_d.dims();
            args.idx = dst_dims_idx;
            ref_post_ops.execute(args);
        }
        
        // 存储结果
        const auto dst_off = dst_d.off_v(dst_dims_idx);
        float *dst_f32 = reinterpret_cast<float *>(dst);
        dst_f32[dst_off] = result;
    });

    return status::success;
}

template struct riscv_matmul_t<data_type::f32>;

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl