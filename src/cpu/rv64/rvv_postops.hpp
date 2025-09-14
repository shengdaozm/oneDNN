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
#ifndef CPU_RV64_RVV_POSTOPS_HPP
#define CPU_RV64_RVV_POSTOPS_HPP

#include <riscv_vector.h>

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {

struct rvv_postops_t {
    rvv_postops_t(const post_ops_t &po)
        : alg_(alg_kind::undef), sum_scale_(0.f), sum_zero_point_(0) {
        if (po.len() > 0) {
            const auto &e = po.entry_[0];
            if (e.is_eltwise()) {
                alg_ = e.eltwise.alg;
            } else if (e.is_sum()) {
                alg_ = alg_kind::sum;
                sum_scale_ = e.sum.scale;
                sum_zero_point_ = e.sum.zero_point;
            }
        }
        assert(po.len() <= 1 && "rvv_postops_t supports at most one post-op");
    }

    static bool post_ops_ok(const post_ops_t &po) {
        if (po.len() == 0) return true;
        if (po.len() > 1) return false;

        const auto &e = po.entry_[0];
        if (e.is_eltwise()) {
            switch (e.eltwise.alg) {
                case alg_kind::eltwise_relu: return true;
                default: return false;
            }
        } else if (e.is_sum()) {
            return true;
        }
        return false;
    }

    inline vfloat32m1_t apply(vfloat32m1_t v, vfloat32m1_t v_old, size_t vl) const {
        switch (alg_) {
            case alg_kind::sum: {
                vfloat32m1_t v_scaled_old = __riscv_vfmv_v_f_f32m1(sum_scale_, vl);
                v_scaled_old = __riscv_vfmul_vv_f32m1(v_scaled_old, v_old, vl);
                return __riscv_vfadd_vv_f32m1(v, v_scaled_old, vl);
            }
            default: return v;
        }
    }

    inline vfloat32m1_t apply(vfloat32m1_t v, size_t vl) const {
        switch (alg_) {
            case alg_kind::eltwise_relu: {
                vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.f, vl);
                return __riscv_vfmax_vv_f32m1(v, zero, vl);
            }
            default: return v;
        }
    }

    inline vfloat32m4_t apply(vfloat32m4_t v, vfloat32m4_t v_old, size_t vl) const {
        switch (alg_) {
            case alg_kind::sum: {
                vfloat32m4_t v_scaled_old = __riscv_vfmv_v_f_f32m4(sum_scale_, vl);
                v_scaled_old = __riscv_vfmul_vv_f32m4(v_scaled_old, v_old, vl);
                return __riscv_vfadd_vv_f32m4(v, v_scaled_old, vl);
            }
            default: return v;
        }
    }

    inline vfloat32m4_t apply(vfloat32m4_t v, size_t vl) const {
        switch (alg_) {
            case alg_kind::eltwise_relu: {
                vfloat32m4_t zero = __riscv_vfmv_v_f_f32m4(0.f, vl);
                return __riscv_vfmax_vv_f32m4(v, zero, vl);
            }
            default: return v;
        }
    }

    inline vint32m2_t apply(vint32m2_t v, vint32m2_t v_old, size_t vl) const {
        switch (alg_) {
            case alg_kind::sum: {
                vfloat32m2_t v_old_f = __riscv_vfcvt_f_x_v_f32m2(v_old, vl);
                vfloat32m2_t zero_point_f = __riscv_vfmv_v_f_f32m2((float)sum_zero_point_, vl);
                v_old_f = __riscv_vfadd_vv_f32m2(v_old_f, zero_point_f, vl);

                vfloat32m2_t sum_scale_f = __riscv_vfmv_v_f_f32m2(sum_scale_, vl);
                v_old_f = __riscv_vfmul_vv_f32m2(v_old_f, sum_scale_f, vl);

                vint32m2_t v_scaled_old_i = __riscv_vfcvt_x_f_v_i32m2(v_old_f, vl);

                return __riscv_vadd_vv_i32m2(v, v_scaled_old_i, vl);
            }
            default: return v;
        }
    }

    inline vint32m2_t apply(vint32m2_t v, size_t vl) const {
        switch (alg_) {
            case alg_kind::eltwise_relu: {
                vint32m2_t zero = __riscv_vmv_v_x_i32m2(0, vl);
                return __riscv_vmax_vv_i32m2(v, zero, vl);
            }
            default: return v;
        }
    }

    inline vint32m4_t apply(vint32m4_t v, vint32m4_t v_old, size_t vl) const {
        switch (alg_) {
            case alg_kind::sum: {
                vfloat32m4_t v_old_f = __riscv_vfcvt_f_x_v_f32m4(v_old, vl);
                vfloat32m4_t zero_point_f = __riscv_vfmv_v_f_f32m4((float)sum_zero_point_, vl);
                v_old_f = __riscv_vfadd_vv_f32m4(v_old_f, zero_point_f, vl);

                vfloat32m4_t sum_scale_f = __riscv_vfmv_v_f_f32m4(sum_scale_, vl);
                v_old_f = __riscv_vfmul_vv_f32m4(v_old_f, sum_scale_f, vl);

                vint32m4_t v_scaled_old_i = __riscv_vfcvt_x_f_v_i32m4(v_old_f, vl);

                return __riscv_vadd_vv_i32m4(v, v_scaled_old_i, vl);
            }
            default: return v;
        }
    }

    inline vint32m4_t apply(vint32m4_t v, size_t vl) const {
        switch (alg_) {
            case alg_kind::eltwise_relu: {
                vint32m4_t zero = __riscv_vmv_v_x_i32m4(0, vl);
                return __riscv_vmax_vv_i32m4(v, zero, vl);
            }
            default: return v;
        }
    }

private:
    alg_kind_t alg_;
    float sum_scale_;
    int sum_zero_point_;
};

} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_RVV_POSTOPS_HPP