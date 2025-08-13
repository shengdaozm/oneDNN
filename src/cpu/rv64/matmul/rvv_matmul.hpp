/*******************************************************************************
* Copyright 2023-2025 Intel Corporation
* Copyright 2023-2025 KNS Group LLC (YADRO)
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

#ifndef CPU_RV64_MATMUL_RVV_MATMUL_HPP
#define CPU_RV64_MATMUL_RVV_MATMUL_HPP

#include "common/c_types_map.hpp"
#include "common/primitive.hpp"
#include "common/utils.hpp"
#include "cpu/matmul/cpu_matmul_pd.hpp"
#include "cpu/platform.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace rv64 {
namespace matmul {

template <data_type_t d_type>
struct riscv_matmul_t : public primitive_t {
    struct pd_t : public ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t {
        using ::dnnl::impl::cpu::matmul::cpu_matmul_pd_t::cpu_matmul_pd_t;

        DECLARE_COMMON_PD_T("RISCV64GCV", riscv_matmul_t);

        status_t init(engine_t *engine) {
            using namespace data_type;
            using smask_t = primitive_attr_t::skip_mask_t;

            bool ok = utils::one_of(src_md()->data_type, f32, bf16, f16, s8)
                    && utils::one_of(weights_md()->data_type, f32, bf16, f16, s8)
                    && utils::one_of(dst_md()->data_type, f32, bf16, f16, s8)
                    && post_ops_ok()
                    && impl::cpu::platform::has_data_type_support(d_type);
            if (!ok) return status::unimplemented;

            return status::success;
        }

    protected:
        bool post_ops_ok() const {
            // TODO: Implement post-ops validation for RVV
            return true;
        }
    };

    riscv_matmul_t(const pd_t *apd) : primitive_t(apd) {}

    status_t execute(const exec_ctx_t &ctx) const override;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd().get(); }
};

} // namespace matmul
} // namespace rv64
} // namespace cpu
} // namespace impl
} // namespace dnnl

#endif // CPU_RV64_MATMUL_RVV_MATMUL_HPP
