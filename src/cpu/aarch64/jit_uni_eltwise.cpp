/*******************************************************************************
* Copyright 2017-2022 Intel Corporation
* Copyright 2021-2023 FUJITSU LIMITED
* Copyright 2022, 2025 Arm Ltd. and affiliates
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

#include "common/bfloat16.hpp"
#include "common/c_types_map.hpp"
#include "common/dnnl_thread.hpp"
#include "common/nstl.hpp"
#include "common/utils.hpp"

#include "cpu/aarch64/jit_generator.hpp"

#include "cpu/aarch64/injectors/jit_uni_eltwise_injector.hpp"
#include "cpu/aarch64/jit_uni_eltwise.hpp"

#define GET_OFF(field) offsetof(jit_args_t, field)

namespace dnnl {
namespace impl {
namespace cpu {
namespace aarch64 {

using namespace Xbyak_aarch64;

struct jit_args_t {
    const void *src; // fwd: src;  bwd: src/dst based on alg;
    const void *dst; // fwd: dst;  bwd: diff_src;
    const void *diff_dst; // fwd: nullptr;  bwd: diff_dst;
    size_t work_amount;
};

struct jit_uni_eltwise_kernel_t : public jit_generator {
    jit_uni_eltwise_kernel_t(const eltwise_pd_t *pd) : pd_(pd) {}

    void operator()(jit_args_t *p) { jit_generator::operator()(p); }

protected:
    const eltwise_pd_t *pd_;

    data_type_t data_type() const {
        return pd_->use_dst() ? pd_->dst_md()->data_type
                              : pd_->src_md()->data_type;
    }
    bool is_bf16() const { return data_type() == data_type::bf16; }
    bool is_f16() const { return data_type() == data_type::f16; }
    int dtype_size() const { return types::data_type_size(data_type()); }
};

// jit kernels
namespace {

template <cpu_isa_t isa>
struct jit_uni_kernel_t : public jit_uni_eltwise_kernel_t {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_kernel)

    jit_uni_kernel_t(const eltwise_pd_t *pd) : jit_uni_eltwise_kernel_t(pd) {
        const auto &desc = *pd_->desc();
        // there's no auxiliary vregs on fwd path
        const bool is_fwd = pd_->is_fwd();
        const bool save_state = is_fwd ? false : true;
        eltwise_injector_.reset(new jit_uni_eltwise_injector_f32<isa>(this,
                desc.alg_kind, desc.alpha, desc.beta, 1.f, save_state,
                reg_injector_table, injector_mask, injector_p_tmp0, is_fwd,
                pd_->use_dst()));
    }

    void generate() override {
        const bool is_fwd = pd_->is_fwd();
        preamble();

        XReg param = param1;
        add_imm(X_TMP_0, param, GET_OFF(src), X_TMP_1);
        ldr(reg_src, ptr(X_TMP_0));
        add_imm(X_TMP_0, param, GET_OFF(dst), X_TMP_1);
        ldr(reg_dst, ptr(X_TMP_0));
        if (!is_fwd) {
            add_imm(X_TMP_0, param, GET_OFF(diff_dst), X_TMP_1);
            ldr(reg_diff_dst, ptr(X_TMP_0));
        }
        add_imm(X_TMP_0, param, GET_OFF(work_amount), X_TMP_1);
        ldr(reg_work_amount, ptr(X_TMP_0));
        eltwise_injector_->load_table_addr();

        // Predicates used for load and store operations.
        // Initially set to ptrue until we have "< vector length"
        // number of items to process.
        ptrue(pg_s.s);
        ptrue(pg_h.h);

        Label tail_predication;
        Label vectorized_loop_start, vectorized_loop_end;

        cmp(reg_work_amount, simd_w());
        b(LT, tail_predication);

        L(vectorized_loop_start);

        // TODO: consider improving.
        // This piece of code is responsible for the preserve_zero function
        // being a natural restriction of this implementation. It works with any
        // dense and blocked layout, but the problem raises when blocking
        // dimension is not divisible by block size. For such case, the code
        // below should save the mask, where zero padding should be preserved
        // and apply it on register before storing into dst memory. Until
        // there's a restriction on certain blocked layouts, when this behavior
        // can be relevantly easy controlled, this will cost much from code
        // perspective and will complicate the compute logic significantly.

        if (is_bf16()) {
            ld1h(vmm_src.h, pg_h / T_z, ptr(reg_src));
            // Convert BF16 input to FP32, apply eltwise op, then convert back to BF16:
            // - unpack BF16 to FP32 by zero-extending
            // - compute eltwise alg in FP32
            // - down convert back to BF16 using bfcvt, and pack result
            mov(tmp0.s, pg_s, vmm_src.s);
            lsl(vmm_src.s, vmm_src.s, 16);
            and_(tmp0.s, 0xFFFF0000);
            eltwise_injector_->compute_vector_range(
                    {vmm_src.getIdx(), tmp0.getIdx()});
            bfcvt(vmm_src.h, pg_h, vmm_src.s);
            bfcvtnt(vmm_src.h, pg_h, tmp0.s);
            st1h(vmm_src.h, pg_h / T_z, ptr(reg_dst));
        } else if (is_f16()) {
            ld1h(vmm_src.h, pg_h / T_z, ptr(reg_src));
            // Convert FP16 to FP32, apply eltwise op, then convert back to FP16:
            // - upcast FP16 to FP32 using fcvt
            // - compute eltwise alg in FP32
            // - downcast FP32 back to FP16 using fcvt, and pack result
            mov(tmp0.s, pg_s, vmm_src.s);
            fcvt(vmm_src.s, pg_h, vmm_src.h);
            // Next two lines could be replaced by fcvtlt(tmp0.s, P_ALL_ONE, tmp0.h)
            // Not currently implemented in xbyak
            lsr(tmp0.s, tmp0.s, 16);
            fcvt(tmp0.s, pg_h, tmp0.h);
            eltwise_injector_->compute_vector_range(
                    {vmm_src.getIdx(), tmp0.getIdx()});
            fcvt(vmm_src.h, pg_s, vmm_src.s);
            // Next three lines could be replaced by fcvtnt(vmm_src.h, P_ALL_ONE, tmp0.s)
            // Not currently implemented in xbyak
            fcvt(tmp0.h, pg_s, tmp0.s);
            lsl(tmp0.s, tmp0.s, 16);
            orr(vmm_src.h, pg_h, tmp0.h);
            st1h(vmm_src.h, pg_h / T_z, ptr(reg_dst));
        } else {
            ld1w(vmm_src.s, pg_s / T_z, ptr(reg_src));
            eltwise_injector_->compute_vector(vmm_src.getIdx());
            if (!is_fwd) {
                ld1w(ZReg(vmm_diff_dst.getIdx()).s, pg_s / T_z,
                        ptr(reg_diff_dst));
                fmul(vmm_src.s, vmm_src.s, vmm_diff_dst);
            }
            st1w(vmm_src.s, pg_s / T_z, ptr(reg_dst));
        }

        const auto shift = vlen();
        add_imm(reg_src, reg_src, shift, X_TMP_0);
        add_imm(reg_dst, reg_dst, shift, X_TMP_0);
        if (!is_fwd) add_imm(reg_diff_dst, reg_diff_dst, shift, X_TMP_0);

        sub_imm(reg_work_amount, reg_work_amount, simd_w(), X_TMP_0);
        cmp(reg_work_amount, simd_w());
        b(GE, vectorized_loop_start);

        L(tail_predication);

        cmp(reg_work_amount, 0);
        b(LE, vectorized_loop_end);

        // Instead of a tail loop, we use SVE predication to only load
        // the remainder elements, with the inactive elements of the vector
        // set to 0. This is done outside of the vectorized_loop to avoid
        // unnecessary overhead.
        mov_imm(X_TMP_1, 0);
        whilelt(pg_s.s, X_TMP_1, reg_work_amount);
        if (is_bf16() || is_f16()) {
            whilelt(pg_h.h, X_TMP_1, reg_work_amount);
        }

        b(vectorized_loop_start);

        L(vectorized_loop_end);

        postamble();

        eltwise_injector_->prepare_table();
    }

private:
    using TReg = typename cpu_isa_traits<isa>::TReg;
    using TRegS = typename cpu_isa_traits<isa>::TRegS;
    int vlen() {
        // TODO: If we do decide to add a different enum for
        // VLA SVE, we should handle this in cpu_isa_traits
        return isa == asimd ? cpu_isa_traits<isa>::vlen : get_sve_length();
    }
    int simd_w() { return vlen() / dtype_size(); }

    XReg reg_src = x11;
    XReg reg_dst = x8;
    XReg reg_injector_table = x9;
    XReg reg_diff_dst = x10;
    XReg reg_work_amount = x6;
    XReg imm_addr64 = x3;
    PReg injector_mask = p1;
    PReg injector_p_tmp0 = p4;
    PReg injector_p_all = p7;

    VReg4S xmm_src {1};
    VReg8H v_bf16 {1};
    VReg8H v_f16 {1};
    TReg vmm_src {1};
    VReg4S xmm_diff_dst {2};
    TRegS vmm_diff_dst {2};
    TReg tmp0 {2};
    std::unique_ptr<jit_uni_eltwise_injector_f32<isa>> eltwise_injector_;

    PReg p_tmp0 {4}; /* Index is temporal. */
    PReg pg_s {5};
    PReg pg_h {7};
};

} // namespace

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper src_d(src_md());

    bool ok = mayiuse(isa) && is_fwd()
            && utils::everyone_is(
                    d_type, src_md()->data_type, dst_md()->data_type)
            && !has_zero_dim_memory() && src_d.is_dense(true)
            && eltwise_injector::is_supported(isa, desc_.alg_kind)
            // refer to a comment in jit_uni_kernel why this is needed
            && IMPLICATION(!src_d.is_dense(), is_zero_preserved())
            && attr()->has_default_values() && set_default_formats_common()
            && src_d == memory_desc_wrapper(dst_md());
    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::jit_uni_eltwise_fwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_fwd_t<isa, d_type>::~jit_uni_eltwise_fwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_fwd_t<isa, d_type>::execute(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, DNNL_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    dst += data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = dst + start;
        args.diff_dst = nullptr;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::pd_t::init(engine_t *engine) {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(data_md());

    bool ok = mayiuse(isa) && !is_fwd()
            && utils::everyone_is(d_type, data_md()->data_type,
                    diff_src_md()->data_type, diff_dst_md()->data_type)
            && !has_zero_dim_memory() && set_default_formats_common()
            && data_d.is_dense(true)
            && eltwise_injector::is_supported(isa, desc_.alg_kind)
            // refer to a comment in jit_uni_kernel why this is needed
            && IMPLICATION(!data_d.is_dense(), is_zero_preserved())
            && data_d == memory_desc_wrapper(diff_dst_md())
            && memory_desc_wrapper(diff_src_md())
                    == memory_desc_wrapper(diff_dst_md())
            && attr()->has_default_values();
    return ok ? status::success : status::unimplemented;
}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::jit_uni_eltwise_bwd_t(const pd_t *apd)
    : primitive_t(apd) {}

template <cpu_isa_t isa, data_type_t d_type>
jit_uni_eltwise_bwd_t<isa, d_type>::~jit_uni_eltwise_bwd_t() = default;

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::init(engine_t *engine) {
    CHECK(safe_ptr_assign(kernel_, new jit_uni_kernel_t<isa>(pd())));
    return kernel_->create_kernel();
}

template <cpu_isa_t isa, data_type_t d_type>
status_t jit_uni_eltwise_bwd_t<isa, d_type>::execute(
        const exec_ctx_t &ctx) const {
    auto src = pd()->use_dst() ? CTX_IN_MEM(const data_t *, DNNL_ARG_DST)
                               : CTX_IN_MEM(const data_t *, DNNL_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, DNNL_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, DNNL_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->data_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const auto nelems = data_d.nelems(true);
    const int simd_w = 64 / data_d.data_type_size();

    src += data_d.offset0();
    diff_dst += diff_data_d.offset0();
    diff_src += diff_data_d.offset0();

    parallel(0, [&](const int ithr, const int nthr) {
        dim_t start {0}, end {0};

        balance211(utils::div_up(nelems, simd_w), nthr, ithr, start, end);
        start = nstl::min(nelems, start * simd_w);
        end = nstl::min(nelems, end * simd_w);
        if (start == end) return;

        jit_args_t args;
        args.src = src + start;
        args.dst = diff_src + start;
        args.diff_dst = diff_dst + start;
        args.work_amount = end - start;
        (*kernel_)(&args);
    });

    return status::success;
}

// Jit uni eltwise is fully vector length agnostic, so we use sve_128
// as alias for VLA SVE.
template struct jit_uni_eltwise_fwd_t<sve_128, data_type::f32>;
template struct jit_uni_eltwise_fwd_t<sve_128, data_type::bf16>;
template struct jit_uni_eltwise_fwd_t<sve_128, data_type::f16>;
template struct jit_uni_eltwise_bwd_t<sve_128, data_type::f32>;

} // namespace aarch64
} // namespace cpu
} // namespace impl
} // namespace dnnl
