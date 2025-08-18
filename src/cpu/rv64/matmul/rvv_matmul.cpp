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
        return status::success;
    }

    // get matmul dimensions
    dnnl::impl::cpu::matmul::matmul_helper_t helper(src_d, weights_d, dst_d);
    const int ndims = pd()->ndims();
    const dim_t M = helper.M();
    const dim_t N = helper.N();
    const dim_t K = helper.K();
    const dim_t batch = helper.batch();

    // Get attribute related data
    const auto &attr = *pd()->attr();
    const bool has_post_ops = !attr.post_ops_.has_default_values();
    // const bool with_scales = !attr.scales_.has_default_values();

    // get mask information for broadcasting
    const int src_mask = utils::get_dims_mask(dst_d.dims(), src_d.dims(), ndims);
    const int wei_mask = utils::get_dims_mask(dst_d.dims(), weights_d.dims(), ndims);
    const int bia_mask = utils::get_dims_mask(dst_d.dims(), bia_d.dims(), ndims);

    // Prepare post ops
    // post_ops_t ref_post_ops;
    // ref_post_ops.prepare(ctx, attr, dst_d.data_type(), dst_d.dims(), ndims);
    
    // rvv_kernel
    auto rvv_matmul_kernel = [&](const dims_t &dst_dims_idx, dim_t m, dim_t n) -> float {
    dims_t src_dims_idx, weights_dims_idx;
    utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
    utils::copy_dims_with_mask(weights_dims_idx, dst_dims_idx, ndims, wei_mask);

    src_dims_idx[ndims - 2] = m;
    src_dims_idx[ndims - 1] = 0; 
    
    weights_dims_idx[ndims - 2] = 0; 
    weights_dims_idx[ndims - 1] = n;

    const float *src_base_ptr = src + src_d.off_v(src_dims_idx);
    const float *weights_base_ptr = weights + weights_d.off_v(weights_dims_idx);
    
    const dim_t weights_k_stride = helper.get_b_stride(ndims - 2);
    
    vfloat32m1_t acc_vec = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());

    for (dim_t k = 0; k < K; ) {
        size_t vl = __riscv_vsetvl_e32m1(K - k);
        
        const float *weights_current_ptr = weights_base_ptr + k * weights_k_stride;

        vfloat32m1_t vec_src = __riscv_vle32_v_f32m1(src_base_ptr + k, vl);
        vfloat32m1_t vec_weights = __riscv_vlse32_v_f32m1(weights_current_ptr, weights_k_stride * sizeof(float), vl);
        
        acc_vec = __riscv_vfmacc_vv_f32m1(acc_vec, vec_src, vec_weights, vl);
        
        k += vl;
    }
    
    vfloat32m1_t sum_vec = __riscv_vfredusum_vs_f32m1_f32m1(acc_vec, __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1()), __riscv_vsetvlmax_e32m1());
    
    return __riscv_vfmv_f_s_f32m1_f32(sum_vec);
};

    // calulate without rvv if K< vlen
    auto scalar_kernel = [&](const dims_t &dst_dims_idx, dim_t m, dim_t n) -> float {
        dims_t src_dims_idx, weights_dims_idx;
        utils::copy_dims_with_mask(src_dims_idx, dst_dims_idx, ndims, src_mask);
        utils::copy_dims_with_mask(weights_dims_idx, dst_dims_idx, ndims, wei_mask);
        
        src_dims_idx[ndims - 2] = m;
        weights_dims_idx[ndims - 1] = n;
        auto &src_k_dim = src_dims_idx[ndims - 1];
        auto &wei_k_dim = weights_dims_idx[ndims - 2];
        
        float acc = 0.0f;
        for (dim_t k = 0; k < K; ++k) {
            src_k_dim = k;
            wei_k_dim = k;
            const auto src_off = src_d.off_v(src_dims_idx);
            const auto weights_off = weights_d.off_v(weights_dims_idx);
            float s = io::load_float_value(data_type::f32, src, src_off);
            float w = io::load_float_value(data_type::f32, weights, weights_off);
            acc += s * w;
        }
        return acc;
    };

    // bias function
    auto ker_bias = [&](const dims_t &dst_dims_idx) -> float {
        dims_t bia_dims_idx;
        utils::copy_dims_with_mask(bia_dims_idx, dst_dims_idx, ndims, bia_mask);
        const auto bias_off = bia_d.off_v(bia_dims_idx);
        return io::load_float_value(data_type::f32, bias, bias_off);
    };

    parallel_nd(batch, M, N, [&](dim_t mb, dim_t m, dim_t n) {
        dims_t dst_dims_idx;
        const size_t l_offset = mb * M * N + m * N + n;
        utils::l_dims_by_l_offset(dst_dims_idx, l_offset, dst_d.dims(), ndims);
        
        float result = (K>=__riscv_vsetvlmax_e32m1()) ? rvv_matmul_kernel(dst_dims_idx, m, n)
                              : scalar_kernel(dst_dims_idx, m, n);
        
        // add bias
        if(bias) {
            result += ker_bias(dst_dims_idx);
        }
        
        // store result
        const auto dst_off = dst_d.off_v(dst_dims_idx);
        io::store_float_value(data_type::f32, result, dst, dst_off);
    });

    return status::success;
}

template struct riscv_matmul_t<data_type::f32>;

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl