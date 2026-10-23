// Axera NPU backend, Qwen3.5 graph template: kernels, IO binding, initialization

#include "ggml-axera-q35.h"

#include <sstream>

// kernel input -> GGUF tensor of the layer, for the inputs taken as they are
static const std::map<std::string, const char *> axera_q35_direct = {
    { "ln1", "attn_norm.weight" },    { "ln2", "post_attention_norm.weight" }, { "w_qkv", "attn_qkv.weight" },
    { "w_z", "attn_gate.weight" },    { "w_b", "ssm_beta.weight" },            { "w_a", "ssm_alpha.weight" },
    { "dt_bias", "ssm_dt.bias" },     { "norm_w", "ssm_norm.weight" },         { "w_out", "ssm_out.weight" },
    { "w_gate", "ffn_gate.weight" },  { "w_up", "ffn_up.weight" },             { "w_down", "ffn_down.weight" },
    { "w_q", "attn_q.weight" },       { "w_k", "attn_k.weight" },              { "w_v", "attn_v.weight" },
    { "q_norm", "attn_q_norm.weight" }, { "k_norm", "attn_k_norm.weight" },    { "w_o", "attn_output.weight" },
};

// MoE layers: the dense MLP inputs come from the shared expert, plus the router and the shared expert's gate
static const std::map<std::string, const char *> axera_q35_moe_direct = {
    { "w_gate", "ffn_gate_shexp.weight" }, { "w_up", "ffn_up_shexp.weight" }, { "w_down", "ffn_down_shexp.weight" },
    { "w_router", "ffn_gate_inp.weight" }, { "w_shg", "ffn_gate_inp_shexp.weight" },
};

// GGUF tensor (name within the layer) of kernel input n, nullptr if none
const char * axera_q35_gguf_name(const axera_q35 & m, const std::string & n) {
    if (n == "ln2" && !m.hybrid) {
        return "ffn_norm.weight";  // Qwen3's name of the norm before the MLP
    }
    if (m.moe) {
        auto it = axera_q35_moe_direct.find(n);
        if (it != axera_q35_moe_direct.end()) {
            return it->second;
        }
    }
    auto it = axera_q35_direct.find(n);
    return it == axera_q35_direct.end() ? nullptr : it->second;
}

// a bf16 view of a GGUF weight for a kernel input of `size` bytes: the tensor itself if it is bf16 in CMM, else a
// cached bf16 copy (quantized GGUF: Q8_0, Q4_K, ...; f16; f32)
AX_ENGINE_IO_BUFFER_T axera_q35_weight_bf16(axera_q35 & m, const ggml_tensor * w, uint32_t size) {
    if (w->type == GGML_TYPE_BF16 && ggml_nbytes(w) == size && ggml_backend_buffer_is_axera(w->buffer)) {
        return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, size);
    }
    if ((uint64_t) ggml_nelements(w) * sizeof(ggml_bf16_t) != size) {
        GGML_ABORT("qwen35 template: %s (%s, %" PRId64 " elements) does not fit a %u-byte bf16 kernel input\n", w->name,
                   ggml_type_name(w->type), ggml_nelements(w), size);
    }
    auto it = m.bf16_of.find(w);
    if (it == m.bf16_of.end()) {
        it = m.bf16_of.emplace(w, axera_to_bf16(w)).first;
    }
    return axera_io(it->second.get(), it->second->vir, size);
}

// kernel input n of `size` bytes from GGUF tensor w: in CMM as it is, a bf16 copy of a quantized / f16 weight, or a
// CMM copy (owned by the layer) of a tensor kept on the CPU
AX_ENGINE_IO_BUFFER_T axera_q35_gguf_input(axera_q35 & m, axera_q35_layer & ly, const ggml_tensor * w,
                                           const std::string & n, uint32_t size) {
    if ((w->type != GGML_TYPE_F32 && w->type != GGML_TYPE_BF16) ||
        (w->type == GGML_TYPE_F32 && ggml_nbytes(w) == 2 * (size_t) size)) {  // quantized / f16 / f32 -> bf16 input
        return axera_q35_weight_bf16(m, w, size);
    }
    if (ggml_nbytes(w) != size) {
        GGML_ABORT("qwen35 template: %s is %zu bytes, kernel input %s wants %u\n", w->name, ggml_nbytes(w), n.c_str(),
                   size);
    }
    if (ggml_backend_buffer_is_axera(w->buffer)) {
        return axera_io((ggml_backend_axera_buffer_context *) w->buffer->context, w->data, size);
    }
    ly.own.push_back(axera_cmm_alloc(size));  // not in CMM (layer kept on the CPU): copy
    auto * b = ly.own.back().get();
    memcpy(b->vir, w->data, size);
    b->flush(b->vir, size);
    return axera_io(b, b->vir, size);
}

// MXFP8 kernel input w_<x> (elements) or s_<x> (scales), quantized once per layer from the GGUF tensor w of w_<x> and
// shared by the layer's decode, prefill and split attention kernels
AX_ENGINE_IO_BUFFER_T axera_q35_mx_input(axera_q35_layer & ly, const ggml_tensor * w, const std::string & n,
                                         uint32_t size) {
    const std::string wn = "w_" + n.substr(2);
    auto              mx = ly.mx.find(wn);
    if (mx == ly.mx.end()) {
        axera_cmm q, sc;
        axera_quantize_mxfp8(w, q, sc);
        mx = ly.mx.emplace(wn, std::make_pair(q.get(), sc.get())).first;
        ly.own.push_back(std::move(q));
        ly.own.push_back(std::move(sc));
    }
    auto * b = n[0] == 'w' ? mx->second.first : mx->second.second;
    if (b->size < size) {
        GGML_ABORT("qwen35 template: %s is %zu bytes, kernel wants %u\n", n.c_str(), b->size, size);
    }
    return axera_io(b, b->vir, size);
}

// the packed variant <name of k>_pk of kernel k (name: k's file name), if built with its layout; GGML_AXERA_PREFILL_PK=0
// keeps the unpacked kernels
static void axera_q35_load_pk(axera_q35_pk_kernel & pk, const std::string & name, ggml_axera_kernel * k) {
    pk = axera_q35_pk_kernel();
    const char * e = std::getenv("GGML_AXERA_PREFILL_PK");
    if (!k || (e && strcmp(e, "0") == 0) || !axera_kernels().has(name + "_pk")) {
        return;
    }
    static const char * packs[4] = { "pw", "ps", "pf", "pb" };
    const std::string   lp = axera_kernels().path(name + "_pk");
    std::ifstream       f(lp.substr(0, lp.size() - strlen(".axmodel")) + ".layout");
    if (!f.good()) {
        GGML_LOG_WARN("%s: %s_pk has no layout file, not used\n", __func__, name.c_str());
        return;
    }
    std::string line;
    while (std::getline(f, line)) {  // "<member> <pack> <offset> <bytes>", the packs as "<pack> - 0 <bytes>"
        std::istringstream ls(line);
        std::string        mn, pn;
        size_t             off = 0, bytes = 0;
        if (!(ls >> mn >> pn >> off >> bytes)) {
            continue;
        }
        const int pi = (int) (std::find(packs, packs + 4, pn == "-" ? mn : pn) - packs);
        GGML_ASSERT(pi < 4);
        if (pn == "-") {
            pk.pack_bytes[pi] = bytes;
            continue;
        }
        const int j = k->input(mn.c_str());
        GGML_ASSERT(j >= 0 && k->info->pInputs[j].nSize == bytes);
        pk.members.push_back({ pi, j, off, bytes });
        pk.sig += line + ";";
    }
    pk.k = axera_kernels().get(name + "_pk");
    for (uint32_t i = 0; i < pk.k->info->nInputSize; ++i) {
        const char * n  = pk.k->info->pInputs[i].pName;
        const int    pi = (int) (std::find(packs, packs + 4, std::string(n)) - packs);
        if (pi < 4 && pk.pack_bytes[pi] == 0) {  // a pack without members (no layout line): bound to a zeroed buffer
            pk.pack_bytes[pi] = pk.k->info->pInputs[i].nSize;
        }
        const int j = pi < 4 ? -1 - pi : k->input(n);
        if (pi == 4 && j < 0) {
            GGML_ABORT("qwen35 template: %s_pk input %s is neither a pack nor an input of %s\n", name.c_str(), n,
                       name.c_str());
        }
        GGML_ASSERT(j < 0 || k->info->pInputs[j].nSize == pk.k->info->pInputs[i].nSize);
        GGML_ASSERT(j >= 0 || pk.k->info->pInputs[i].nSize == pk.pack_bytes[pi]);
        pk.in.push_back(j);
    }
    for (uint32_t i = 0; i < pk.k->info->nOutputSize; ++i) {
        const int j = k->output(pk.k->info->pOutputs[i].pName);
        GGML_ASSERT(j >= 0 && k->info->pOutputs[j].nSize == pk.k->info->pOutputs[i].nSize);
        pk.out.push_back(j);
    }
}

// binds one layer's kernel IO: GGUF tensors in CMM directly where the layout matches, converted copies otherwise
void axera_q35_bind_layer(axera_q35 & m, int il, const std::map<std::string, ggml_tensor *> & t) {
    auto get = [&](const char * name) -> ggml_tensor * {
        char key[96];
        snprintf(key, sizeof(key), "blk.%d.%s", il + m.off, name);
        auto it = t.find(key);
        if (it == t.end()) {
            GGML_ABORT("qwen35 template: tensor %s not in the graph\n", key);
        }
        return it->second;
    };
    // llama.cpp recurrent cache of this layer (cache_r_l<il> / cache_s_l<il>), if one sequence row in CMM holds the state
    auto cache = [&](const char * which, uint32_t size) -> ggml_tensor * {
        auto it = t.find("cache_" + std::string(which) + "_l" + std::to_string(il + m.off));
        if (it == t.end() || !ggml_backend_buffer_is_axera(it->second->buffer) || it->second->nb[1] < size) {
            return nullptr;
        }
        return it->second;
    };
    axera_q35_layer & ly = m.layers[il];
    ly.gdn               = t.count("blk." + std::to_string(il + m.off) + ".ssm_a") > 0;
    ly.own.clear();
    ly.state.clear();
    ly.mx.clear();
    ly.in_place = false;
    ly.cache_state.clear();
    ly.row = 0;
    ly.kv.clear();
    ly.stream = 0;
    ly.fd.clear();
    if (!ly.gdn && m.flash) {  // split attention: axera_q35_bind_flash
        ly.in.clear();
        ly.out.clear();
        return;
    }
    ggml_axera_kernel * k = ly.gdn ? m.gdn : m.attn;
    ly.in.assign(k->info->nInputSize, {});
    ly.out.assign(k->info->nOutputSize, {});

    // layer input/output ping-pong: even layers read A and write B, odd layers the other way round
    auto * xin  = (il % 2 == 0 ? m.A : m.B).get();
    auto * xout = (il % 2 == 0 ? m.B : m.A).get();
    auto own = [&](uint32_t size) {
        ly.own.push_back(axera_cmm_alloc(size));
        return ly.own.back().get();
    };

    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        ggml_backend_axera_buffer_context * b = nullptr;
        void *                              p = nullptr;
        if (n == "x") {
            b = xin;
        } else if (n == "pos") {
            b = m.POS.get();
        } else if (n == "nblk") {
            b = m.NBLK.get();
        } else if (n == "mask") {
            b = m.MASK.get();
        } else if (n == "cos") {
            b = m.COS.get();
        } else if (n == "sin") {
            b = m.SIN.get();
        } else if ((n == "state_in" || n == "conv_in") && cache(n == "state_in" ? "s" : "r", size)) {
            // in place in the llama.cpp recurrent cache: its checkpoints save and restore exactly this state
            ggml_tensor * c = cache(n == "state_in" ? "s" : "r", size);
            b               = (ggml_backend_axera_buffer_context *) c->buffer->context;
            p               = c->data;
            ly.in_place     = true;
            axera_q35_layer::region rg;
            rg.b      = b;
            rg.base   = (char *) c->data;
            rg.size   = size;
            rg.stride = c->nb[1];
            rg.in     = (int) i;
            rg.conv   = n == "conv_in";
            ly.cache_state.push_back(rg);
        } else if (n == "state_in" || n == "conv_in") {
            b = own(size);
            ly.state.push_back(b);
            (n == "state_in" ? ly.si : ly.ci) = (int) i;
        } else if (n == "a_log") {  // GGUF ssm_a = -exp(A_log)
            const ggml_tensor * a = get("ssm_a");
            GGML_ASSERT(a->type == GGML_TYPE_F32 && ggml_nbytes(a) == size);
            b = own(size);
            for (int64_t j = 0; j < ggml_nelements(a); ++j) {
                ((float *) b->vir)[j] = logf(-((const float *) a->data)[j]);
            }
            b->flush(b->vir, size);
        } else if (n == "conv_w") {  // GGUF f32 [channels][taps] -> bf16, taps reversed (kernel tap 0 = current token)
            const ggml_tensor * c = get("ssm_conv1d.weight");
            const int64_t       taps = c->ne[0], ch = c->ne[1];
            GGML_ASSERT(c->type == GGML_TYPE_F32 && ch * taps * (int64_t) sizeof(ggml_bf16_t) == size);
            b = own(size);
            for (int64_t r = 0; r < ch; ++r) {
                for (int64_t j = 0; j < taps; ++j) {
                    ((ggml_bf16_t *) b->vir)[r * taps + j] =
                        ggml_fp32_to_bf16(((const float *) c->data)[r * taps + taps - 1 - j]);
                }
            }
            b->flush(b->vir, size);
        } else if ((n.rfind("w_", 0) == 0 || n.rfind("s_", 0) == 0) && k->input(("s_" + n.substr(2)).c_str()) >= 0) {
            // MXFP8 kernel: w_<x> elements and s_<x> scales, quantized from the GGUF tensor of w_<x>
            const char * g = axera_q35_gguf_name(m, "w_" + n.substr(2));
            if (!g) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            ly.in[i] = axera_q35_mx_input(ly, get(g), n, size);
            continue;
        } else {
            const char * g = axera_q35_gguf_name(m, n);
            if (!g) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            ly.in[i] = axera_q35_gguf_input(m, ly, get(g), n, size);
            continue;
        }
        if (b->size < size && p == nullptr) {
            GGML_ABORT("qwen35 template: buffer for %s too small\n", n.c_str());
        }
        ly.in[i] = axera_io(b, p ? p : b->vir, size);
    }
    for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
        const std::string n    = k->info->pOutputs[i].pName;
        const uint32_t    size = k->info->pOutputs[i].nSize;
        ggml_backend_axera_buffer_context * b;
        if (n == "out") {
            b = m.moe ? m.BASE.get() : xout;  // MoE: the experts kernel adds the routed experts and writes xout
        } else if (n == "h2_out" || n == "logits_out") {
            b = (n == "h2_out" ? m.H2 : m.LOGITS).get();
        } else if (m.kv_llama && (n == "kc" || n == "vc")) {
            // the llama.cpp KV cache of this layer: [cells][heads x dim] f16, the layout of the _kvc kernels
            // N cells per stream, allocated with room for LMAX after the last stream (the IO size must be the
            // kernel's); the kernels only touch the blocks below the runtime block count, all within the stream
            ggml_tensor * c = t.at("cache_" + std::string(n == "kc" ? "k" : "v") + "_l" + std::to_string(il + m.off));
            GGML_ASSERT(c->nb[1] * m.LMAX == size &&
                        ggml_backend_buft_get_alloc_size(c->buffer->buft, c) >= (c->ne[2] - 1) * c->nb[2] + size);
            axera_q35_layer::kv_region kr;
            kr.b      = (ggml_backend_axera_buffer_context *) c->buffer->context;
            kr.base   = (char *) c->data;
            kr.size   = size;
            kr.stride = c->nb[2];
            kr.row    = c->nb[1];
            kr.out    = (int) i;
            kr.v      = n == "vc";
            ly.kv.push_back(kr);
            ly.out[i] = axera_io(kr.b, kr.base, size);
            continue;
        } else if (ly.in_place && (n == "state_out" || n == "conv_out")) {
            const int j = k->input(n == "state_out" ? "state_in" : "conv_in");
            ly.out[i]   = ly.in[j];  // same buffer as the input
            for (auto & rg : ly.cache_state) {
                if (rg.in == j) {
                    rg.out = (int) i;
                }
            }
            continue;
        } else {  // state_out / conv_out, or the resident kc / vc
            b = own(size);
            if (n == "state_out" || n == "conv_out") {
                ly.state.push_back(b);
                (n == "state_out" ? ly.so : ly.co) = (int) i;
            }
        }
        ly.out[i] = axera_io(b, b->vir, size);
    }
}

// prefill IO of one layer for chunk size v: the decode IO entries of the same names (weights, recurrent state, KV
// cache), with P-row activations, the chunk mask and the chunk's RoPE tables
void axera_q35_bind_prefill(axera_q35 & m, int il, size_t v, const std::map<std::string, ggml_tensor *> & t) {
    axera_q35_layer &        ly = m.layers[il];
    axera_q35::prefill &     pf = m.pre[v];
    if ((!ly.gdn && (m.flash || !pf.attn)) || (ly.gdn && !pf.gdn)) {
        // split attention: axera_q35_bind_flash
        ly.pre.resize(m.pre.size());
        ly.pre[v].batch = pf.batch;
        return;
    }
    ggml_axera_kernel *      k  = ly.gdn ? pf.gdn : pf.attn;
    ggml_axera_kernel *      kd = ly.gdn ? m.gdn : m.attn;
    auto *                   xin  = (il % 2 == 0 ? pf.A : pf.B).get();
    auto *                   xout = (il % 2 == 0 ? pf.B : pf.A).get();
    ly.pre.resize(m.pre.size());
    axera_q35_layer::prefill_io & io = ly.pre[v];
    io.in.assign(k->info->nInputSize, {});
    io.out.assign(k->info->nOutputSize, {});
    io.batch = pf.batch;
    for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
        const std::string n    = k->info->pInputs[i].pName;
        const uint32_t    size = k->info->pInputs[i].nSize;
        if (pf.CPOS && (n == "pos" || n == "nblk")) {
            auto * b = (n == "pos" ? pf.CPOS : pf.CNBLK).get();
            io.in[i] = axera_io(b, b->vir, size);
            continue;
        }
        if (n == "x" || n == "mask" || n == "cos" || n == "sin" || (pf.batch && n == "pos")) {
            auto * b  = n == "x" ? xin : n == "mask" ? pf.MASK.get() : n == "cos" ? pf.COS.get() : n == "sin" ? pf.SIN.get() :
                                                                                                    pf.POS.get();
            io.in[i]  = axera_io(b, b->vir, size);
            continue;
        }
        if (pf.batch && (n == "state_in" || n == "conv_in")) {
            // P consecutive rows of the llama.cpp recurrent cache, from row 0 until axera_q35_set_batch_rows
            const int j = kd->input(n.c_str());
            for (auto & rg : ly.cache_state) {
                if (rg.in == j) {
                    GGML_ASSERT(size == pf.P * rg.stride);
                    io.in[i] = axera_io(rg.b, rg.base, size);
                }
            }
            (n == "state_in" ? io.si : io.ci) = (int) i;
            continue;
        }
        const int j = kd->input(n.c_str());
        if (j < 0 || kd->info->pInputs[j].nSize != size) {
            // a bf16 prefill weight next to MXFP8 decode kernels: the GGUF tensor itself
            const char * g  = axera_q35_gguf_name(m, n);
            auto         it = !g ? t.end() : t.find("blk." + std::to_string(il + m.off) + "." + g);
            if (it == t.end()) {
                GGML_ABORT("qwen35 template: prefill input %s matches neither the decode kernel nor a GGUF tensor\n",
                           n.c_str());
            }
            io.in[i] = axera_q35_weight_bf16(m, it->second, size);
            continue;
        }
        io.in[i] = ly.in[j];
        if (n == "state_in") {
            io.si = (int) i;
        } else if (n == "conv_in") {
            io.ci = (int) i;
        }
    }
    for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
        const std::string n = k->info->pOutputs[i].pName;
        if (n == "topk_v" || n == "topk_i") {  // the router's top-K (kernels built with it)
            auto * b  = (n == "topk_v" ? pf.TKV : pf.TKI).get();
            io.out[i] = axera_io(b, b->vir, k->info->pOutputs[i].nSize);
            continue;
        }
        if (n == "out" || n == "h2_out" || n == "logits_out") {  // MoE: the experts add to the base, into xout
            auto * b  = n == "h2_out" ? pf.H2.get() : n == "logits_out" ? pf.LOG.get() : m.moe ? pf.BASE.get() : xout;
            io.out[i] = axera_io(b, b->vir, k->info->pOutputs[i].nSize);
            continue;
        }
        if (pf.batch && (n == "state_out" || n == "conv_out")) {  // in place
            const int si = n == "state_out" ? io.si : io.ci;
            GGML_ASSERT(si >= 0);
            io.out[i] = io.in[si];
            (n == "state_out" ? io.so : io.co) = (int) i;
            continue;
        }
        const int j = kd->output(n.c_str());
        GGML_ASSERT(j >= 0 && kd->info->pOutputs[j].nSize == k->info->pOutputs[i].nSize);
        io.out[i] = ly.out[j];
        if (n == "state_out") {
            io.so = (int) i;
        } else if (n == "conv_out") {
            io.co = (int) i;
        } else if (n == "kc") {
            io.ko = (int) i;
        } else if (n == "vc") {
            io.vo = (int) i;
        }
    }
    // the packed variant: the layer's packs (one set per layout, shared by the chunk sizes and group copies), filled
    // once from the unpacked kernel's bound weights; the packs bound here, the other IO per call (run_layer)
    if (const auto & pk = ly.gdn ? pf.gdn_pk : pf.attn_pk; pk.k) {
        auto & packs = ly.pk_packs[pk.sig];
        if (!packs[0] && !packs[1] && !packs[2] && !packs[3]) {
            for (int pi = 0; pi < 4; ++pi) {
                if (pk.pack_bytes[pi]) {
                    packs[pi] = axera_cmm_alloc(pk.pack_bytes[pi]);
                    memset(packs[pi]->vir, 0, pk.pack_bytes[pi]);
                }
            }
            for (const auto & mm : pk.members) {
                const AX_ENGINE_IO_BUFFER_T & src = io.in[mm.body];
                GGML_ASSERT(src.pVirAddr && src.nSize == mm.bytes && mm.off + mm.bytes <= pk.pack_bytes[mm.pack]);
                memcpy((char *) packs[mm.pack]->vir + mm.off, src.pVirAddr, mm.bytes);
            }
            for (auto & b : packs) {
                if (b) {
                    b->flush(b->vir, b->size);
                }
            }
        }
        io.pin.assign(pk.in.size(), {});
        io.pout.assign(pk.out.size(), {});
        for (size_t i = 0; i < pk.in.size(); ++i) {
            if (pk.in[i] < 0) {
                auto & b  = packs[-1 - pk.in[i]];
                io.pin[i] = axera_io(b.get(), b->vir, (uint32_t) pk.pack_bytes[-1 - pk.in[i]]);
            }
        }
    }
    if (ly.gdn && pf.gdn_rs) {  // the snapshot kernel: the same inputs and outputs by name, plus the snapshots
        ggml_axera_kernel * kr = pf.gdn_rs;
        io.rin.assign(kr->info->nInputSize, {});
        io.rout.assign(kr->info->nOutputSize, {});
        io.rs_state.assign(pf.P, -1);
        io.rs_conv.assign(pf.P, -1);
        for (uint32_t i = 0; i < kr->info->nInputSize; ++i) {
            const std::string n = kr->info->pInputs[i].pName;
            const int         j = k->input(n.c_str());
            GGML_ASSERT(j >= 0 && k->info->pInputs[j].nSize == kr->info->pInputs[i].nSize);
            io.rin[i] = io.in[j];
            io.rsi    = n == "state_in" ? (int) i : io.rsi;
            io.rci    = n == "conv_in" ? (int) i : io.rci;
        }
        for (uint32_t i = 0; i < kr->info->nOutputSize; ++i) {
            const std::string n = kr->info->pOutputs[i].pName;
            if (n.rfind("state_rs", 0) == 0 || n.rfind("conv_rs", 0) == 0) {
                const bool st = n[0] == 's';
                const int  sl = atoi(n.c_str() + (st ? 8 : 7));
                GGML_ASSERT(sl >= 1 && sl < pf.P);
                (st ? io.rs_state : io.rs_conv)[sl] = (int) i;
                continue;
            }
            const int j = k->output(n.c_str());
            GGML_ASSERT(j >= 0 && k->info->pOutputs[j].nSize == kr->info->pOutputs[i].nSize);
            io.rout[i] = io.out[j];
            io.rso     = n == "state_out" ? (int) i : io.rso;
            io.rco     = n == "conv_out" ? (int) i : io.rco;
        }
        GGML_ASSERT(io.rsi >= 0 && io.rci >= 0 && io.rso >= 0 && io.rco >= 0);
    }
}

// LMAX and BC of an attention kernel, from its mask input [LMAX / BC, BC]
void axera_q35_context(ggml_axera_kernel * k, int & LMAX, int & BC) {
    const int i = k->input("mask");
    GGML_ASSERT(i >= 0);
    const AX_ENGINE_IOMETA_T & mt = k->info->pInputs[i];
    if (mt.nShapeSize != 2 || mt.nSize != (uint32_t) (mt.pShape[0] * mt.pShape[1]) * sizeof(ggml_bf16_t)) {
        GGML_ABORT("qwen35 template: attention kernel mask input is not [LMAX / BC, BC] bf16\n");
    }
    LMAX = mt.pShape[0] * mt.pShape[1];
    BC   = mt.pShape[1];
}

// the split attention kernels for R rows (qwen35_attn_fd_head<R>, _seg<R>_c<S>, _tail<R>) and their buffers
bool axera_q35_load_flash(axera_q35 & m, int R, bool seg_only) {
    if (m.fd.count(R)) {
        return seg_only || m.fd.at(R).head != nullptr;
    }
    const std::string r  = std::to_string(R);
    // MoE: the tail ends with the MoE's dense part (base, h2, router logits), qwen35moe_attn_fd_tail<R>
    std::string hn = m.ap + "_attn_fd_head" + r, tn = (m.moe ? "qwen35moe" : m.ap) + std::string("_attn_fd_tail") + r;
    std::string       sn = m.ap + "_attn_fd_seg" + r + "_c" + std::to_string(m.S);
    // head / tail in the run's weight format where those kernels exist: MXFP8 for decode with MXFP8 weights, for
    // prefill rows with the MXFP8 prefill kernels
    if (axera_weights_mx8() && (R == 1 || m.pre_mx) && std::ifstream(axera_kernels().path(hn + "_mx8")).good() &&
        std::ifstream(axera_kernels().path(tn + "_mx8")).good()) {
        hn += "_mx8";
        tn += "_mx8";
    }
    const bool ht = std::ifstream(axera_kernels().path(hn)).good() && std::ifstream(axera_kernels().path(tn)).good();
    if (!std::ifstream(axera_kernels().path(sn)).good() || (!ht && !seg_only)) {
        return false;
    }
    axera_q35::fd_kernels fk;
    fk.head = ht ? axera_kernels().get(hn) : nullptr;
    fk.seg  = axera_kernels().get(sn);
    fk.tail = ht ? axera_kernels().get(tn) : nullptr;
    const AX_ENGINE_IOMETA_T & o = fk.seg->info->pOutputs[fk.seg->output("o_out")];
    GGML_ASSERT(o.nShapeSize == 3 || o.nShapeSize == 4);
    const int lead = o.nShapeSize - 3;  // [cores] [hkv][gr][d]
    fk.cores = lead ? o.pShape[0] : 1;
    fk.hkv   = o.pShape[lead];
    fk.gr    = o.pShape[lead + 1];
    fk.d     = o.pShape[lead + 2];
    GGML_ASSERT(axera_input_size(fk.seg, "nblk") == (size_t) (fk.cores > 1 ? 2 * fk.cores : 1) * sizeof(int32_t));
    fk.BLK  = axera_cmm_alloc(axera_input_size(fk.seg, "nblk"));
    const AX_ENGINE_IOMETA_T & mt = fk.seg->info->pInputs[fk.seg->input("mask")];
    GGML_ASSERT(mt.nShapeSize == 3 && mt.pShape[1] == R && (int64_t) mt.pShape[0] * mt.pShape[2] == m.S);
    fk.bc   = mt.pShape[2];
    // head outputs: q [R][HQ][D] bf16 (the segment input), gate likewise, new K / V rows [R][HKV * D] f16
    const size_t qsz = axera_input_size(fk.seg, "qs"), kvsz = (size_t) R * fk.hkv * fk.d * sizeof(ggml_fp16_t);
    GGML_ASSERT(!fk.head || (axera_output_size(fk.head, "q_out") == qsz && axera_output_size(fk.head, "k_out") == kvsz));
    fk.Q    = axera_cmm_alloc(qsz);
    fk.GATE = axera_cmm_alloc(qsz);
    fk.K    = axera_cmm_alloc(kvsz);
    fk.V    = axera_cmm_alloc(kvsz);
    fk.M    = axera_cmm_alloc(axera_output_size(fk.seg, "m_out"));
    fk.Lsum = axera_cmm_alloc(axera_output_size(fk.seg, "l_out"));
    fk.O    = axera_cmm_alloc(axera_output_size(fk.seg, "o_out"));
    fk.ATT  = fk.tail ? axera_cmm_alloc(axera_input_size(fk.tail, "attn")) : nullptr;
    m.fd.emplace(R, std::move(fk));
    return true;
}

// the split attention kernels and state of run kind v (-1: decode, else m.pre[v]): by rows, a group copy by fdk
int axera_q35_fd_key(const axera_q35 & m, int v) {
    return v < 0 ? 1 : m.pre[v].fdk ? m.pre[v].fdk : m.pre[v].P;
}

// a copy of the split attention kernels for R rows under key: same kernels, buffers of its own
void axera_q35_clone_flash(axera_q35 & m, int R, int key) {
    const axera_q35::fd_kernels & a = m.fd.at(R);
    axera_q35::fd_kernels         fk;
    fk.head  = a.head;
    fk.seg   = a.seg;
    fk.tail  = a.tail;
    fk.hkv   = a.hkv;
    fk.gr    = a.gr;
    fk.d     = a.d;
    fk.bc    = a.bc;
    fk.cores = a.cores;
    for (auto [dst, src] : { std::pair(&fk.Q, &a.Q), std::pair(&fk.GATE, &a.GATE), std::pair(&fk.K, &a.K),
                             std::pair(&fk.V, &a.V), std::pair(&fk.M, &a.M), std::pair(&fk.Lsum, &a.Lsum),
                             std::pair(&fk.O, &a.O), std::pair(&fk.ATT, &a.ATT), std::pair(&fk.BLK, &a.BLK) }) {
        *dst = *src ? axera_cmm_alloc((*src)->size) : nullptr;
    }
    m.fd[key] = std::move(fk);
}

// split attention IO of attention layer il for run kind v (-1: decode, else m.pre[v]): head and tail on the run's
// activations and RoPE tables and the GGUF weights; the segment kernel's KV and mask inputs are rebound per call
void axera_q35_bind_flash(axera_q35 & m, int il, int v, const std::map<std::string, ggml_tensor *> & t) {
    axera_q35_layer &       ly = m.layers[il];
    axera_q35::fd_kernels & fk = m.fd.at(axera_q35_fd_key(m, v));
    auto & A   = v < 0 ? m.A : m.pre[v].A;
    auto & B   = v < 0 ? m.B : m.pre[v].B;
    auto * xin = (il % 2 == 0 ? A : B).get(), *xout = (il % 2 == 0 ? B : A).get();
    ly.fd.resize(m.pre.size() + 1);
    axera_q35_layer::fd_io & io = ly.fd[v + 1];
    if (ly.kv.empty()) {  // the layer's KV cache: the host writes new cells, the segment kernel reads S cells
        for (const char * kv : { "k", "v" }) {
            const ggml_tensor * c = t.at("cache_" + std::string(kv) + "_l" + std::to_string(il + m.off));
            const int64_t       n = (m.N + m.S - 1) / m.S * m.S;
            GGML_ASSERT(ggml_backend_buft_get_alloc_size(c->buffer->buft, c) >= (c->ne[2] - 1) * c->nb[2] + n * c->nb[1]);
            axera_q35_layer::kv_region kr;
            kr.b      = (ggml_backend_axera_buffer_context *) c->buffer->context;
            kr.base   = (char *) c->data;
            kr.size   = m.S * c->nb[1];
            kr.stride = c->nb[2];
            kr.row    = c->nb[1];
            kr.v      = kv[0] == 'v';
            ly.kv.push_back(kr);
        }
    }
    using buf = std::pair<ggml_backend_axera_buffer_context *, char *>;
    auto bind = [&](ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in, std::vector<AX_ENGINE_IO_BUFFER_T> & out,
                    const std::map<std::string, buf> & bufs) {
        in.assign(k->info->nInputSize, {});
        out.assign(k->info->nOutputSize, {});
        for (uint32_t i = 0; i < k->info->nInputSize; ++i) {
            const std::string n    = k->info->pInputs[i].pName;
            const uint32_t    size = k->info->pInputs[i].nSize;
            auto              b    = bufs.find(n);
            if (b != bufs.end()) {
                in[i] = axera_io(b->second.first, b->second.second, size);
                continue;
            }
            const bool mx = (n.rfind("w_", 0) == 0 || n.rfind("s_", 0) == 0) && k->input(("s_" + n.substr(2)).c_str()) >= 0;
            const char * g  = axera_q35_gguf_name(m, mx ? "w_" + n.substr(2) : n);
            auto         it = !g ? t.end() : t.find("blk." + std::to_string(il + m.off) + "." + g);
            if (it == t.end()) {
                GGML_ABORT("qwen35 template: kernel input %s has no GGUF tensor\n", n.c_str());
            }
            in[i] = mx ? axera_q35_mx_input(ly, it->second, n, size) : axera_q35_gguf_input(m, ly, it->second, n, size);
        }
        for (uint32_t i = 0; i < k->info->nOutputSize; ++i) {
            const auto & b = bufs.at(k->info->pOutputs[i].pName);
            out[i]         = axera_io(b.first, b.second, k->info->pOutputs[i].nSize);
        }
    };
    auto at   = [](const axera_cmm & c) { return buf(c.get(), (char *) c->vir); };
    auto & cs = v < 0 ? m.COS : m.pre[v].COS;
    auto & sn = v < 0 ? m.SIN : m.pre[v].SIN;
    bind(fk.head, io.hin, io.hout,
         { { "x", buf(xin, (char *) xin->vir) }, { "cos", at(cs) }, { "sin", at(sn) }, { "q_out", at(fk.Q) },
           { "gate_out", at(fk.GATE) }, { "k_out", at(fk.K) }, { "v_out", at(fk.V) } });
    const auto & kk = ly.kv[0].v ? ly.kv[1] : ly.kv[0];
    const auto & vv = ly.kv[0].v ? ly.kv[0] : ly.kv[1];
    bind(fk.seg, io.sin, io.sout,
         { { "qs", at(fk.Q) }, { "kc", buf(kk.b, kk.base) }, { "vc", buf(vv.b, vv.base) }, { "nblk", at(fk.BLK) },
           { "mask", at(fk.Q) }, { "m_out", at(fk.M) }, { "l_out", at(fk.Lsum) }, { "o_out", at(fk.O) } });
    io.kc   = fk.seg->input("kc");
    io.vc   = fk.seg->input("vc");
    io.mask = fk.seg->input("mask");
    std::map<std::string, buf> tb = { { "x", buf(xin, (char *) xin->vir) }, { "attn", at(fk.ATT) },
                                      { "gate", at(fk.GATE) }, { "out", buf(xout, (char *) xout->vir) } };
    if (m.moe) {  // the experts add to the base, into xout
        tb["out"]        = at(v < 0 ? m.BASE : m.pre[v].BASE);
        tb["h2_out"]     = at(v < 0 ? m.H2 : m.pre[v].H2);
        tb["logits_out"] = at(v < 0 ? m.LOGITS : m.pre[v].LOG);
    }
    bind(fk.tail, io.tin, io.tout, tb);
}

// experts [n][codes | scales] -> planar, in place: all n experts' codes, then all their scales (the whole-layer
// kernels pick an expert out of each plane at run time); once per tensor (contexts of a model share it)
static void axera_mxfp4_planar(ggml_backend_axera_buffer_context * b, char * base, int64_t n, size_t data, size_t scale) {
    static std::set<const void *> done;
    if (!done.insert(base).second) {
        return;
    }
    const size_t      stride = data + scale;
    std::vector<char> tmp((size_t) n * stride);
    memcpy(tmp.data(), base, tmp.size());
    axera_host_pool().run(n, [&](int64_t e) {
        memcpy(base + e * data, tmp.data() + e * stride, data);
        memcpy(base + n * data + e * scale, tmp.data() + e * stride + data, scale);
    });
    b->flush(base, tmp.size());
}

// the MoE kernels read MXFP4 experts in place, repacked: experts in the GGUF file mapping (mmap, read only) get a CMM
// copy, made once per tensor and shared by the model's contexts (kept like the repack bookkeeping, by address). A
// layer's gate and up (or gate-up) go back to back into one buffer as llama.cpp allocates them, down into another
static void axera_q35_experts_to_cmm(const ggml_tensor * w[3]) {
    static std::mutex                                 mu;
    static std::map<const ggml_tensor *, ggml_tensor> copy;
    std::lock_guard<std::mutex>                       lock(mu);
    for (auto [g0, g1] : { std::pair<int, int>{ 0, 2 }, std::pair<int, int>{ 2, 3 } }) {
        if (w[g0]->buffer == nullptr || w[g0]->buffer->buft != ggml_backend_axera_mapped_buffer_type()) {
            continue;
        }
        if (!copy.count(w[g0])) {
            size_t bytes = 0;
            for (int g = g0; g < g1; ++g) {
                bytes += w[g] ? ggml_nbytes(w[g]) : 0;
            }
            ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(ggml_backend_axera_buffer_type(), bytes);
            if (!buf) {
                GGML_ABORT("qwen35 template: no CMM for a copy of %s (%zu bytes)\n", w[g0]->name, bytes);
            }
            ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            char * p = (char *) ggml_backend_buffer_get_base(buf);
            for (int g = g0; g < g1; ++g) {
                if (w[g]) {
                    ggml_tensor c = *w[g];
                    c.buffer      = buf;
                    c.data        = p;
                    c.view_src    = nullptr;
                    ggml_backend_tensor_set(&c, w[g]->data, 0, ggml_nbytes(w[g]));
                    copy.emplace(w[g], c);
                    p += ggml_nbytes(w[g]);
                }
            }
        }
        for (int g = g0; g < g1; ++g) {
            w[g] = w[g] ? &copy.at(w[g]) : nullptr;
        }
    }
}

// a packed kernel's member: the layer kernel's bound input into its pack (column blocks: each core's columns contiguous)
static void axera_q35_pack_member(const axera_q35_layer & ly, const axera_q35::full_kernel::member & mm,
                                  const axera_cmm & pk, size_t pack_bytes) {
    const AX_ENGINE_IO_BUFFER_T & src = ly.in[mm.body];
    GGML_ASSERT(src.pVirAddr && src.nSize == mm.bytes && mm.off + mm.bytes <= pack_bytes);
    char *       dst = (char *) pk->vir + mm.off;
    const char * sp  = (const char *) src.pVirAddr;
    if (mm.blocks.empty()) {
        memcpy(dst, sp, mm.bytes);
        return;
    }
    GGML_ASSERT(mm.rows * mm.row_bytes == mm.bytes);
    for (const auto & [x, y] : mm.blocks) {
        for (size_t r = 0; r < mm.rows; ++r, dst += y - x) {
            memcpy(dst, sp + r * mm.row_bytes + x, y - x);
        }
    }
}

void axera_q35_init(axera_q35 & m, int L, int H, const std::map<std::string, ggml_tensor *> & t,
                    const ggml_tensor * lm, const ggml_tensor * rope) {
    m.L    = L;
    m.H    = H;
    const bool mx8 = axera_weights_mx8();
    m.moe          = t.count("blk." + std::to_string(m.off) + ".ffn_gate_inp.weight") > 0;  // Qwen3.5 MoE: qwen35moe_* kernels
    // Qwen3.5 or Qwen3 by the layers (the MTP graph holds the Qwen3.5 NextN attention block alone)
    m.hybrid = m.mtp || m.moe;
    for (int il = 0; il < L && !m.hybrid; ++il) {
        m.hybrid = t.count("blk." + std::to_string(il + m.off) + ".ssm_a") > 0;
    }
    m.ap                 = m.hybrid ? "qwen35" : "qwen3";
    const std::string kp = m.moe ? "qwen35moe" : m.ap;
    // RoPE of the graph: rotated dims, base and linear position scale (YaRN is not implemented by the template)
    int32_t rp[16];
    memcpy(rp, rope->op_params, sizeof(rp));
    float base, scale, ext;
    memcpy(&base, rp + 5, sizeof(float));
    memcpy(&scale, rp + 6, sizeof(float));
    memcpy(&ext, rp + 7, sizeof(float));
    if (ext != 0.0f) {
        GGML_ABORT("qwen35 template: YaRN RoPE scaling is not supported\n");
    }
    m.ROT        = rp[1];
    m.THETA      = base;
    m.freq_scale = scale;
    m.rope_mode  = rp[2];
    memcpy(m.sections, rp + 11, sizeof(m.sections));
    // the llama.cpp KV cache is usable if every attention layer's cache is f16 cells of [kv heads x head dim] in CMM
    // (N cells, padded at allocation to the context LMAX of the smallest qwen35_attn*_kvc_c<LMAX> kernel that holds
    // them), V not transposed (flash-attention layout)
    const std::string attn_name = kp + (mx8 ? "_attn_mx8" : "_attn");
    {
        auto ki = t.find("attn_inp_k_idxs"), vi = t.find("attn_inp_v_idxs");
        m.kv_llama = ki != t.end() && vi != t.end() && ki->second->ne[0] == vi->second->ne[0];
    }
    m.N = 0;
    for (int il = 0; il < L && m.kv_llama; ++il) {
        if (t.count("blk." + std::to_string(il + m.off) + ".ssm_a")) {
            continue;
        }
        for (const char * kv : { "k", "v" }) {
            auto it = t.find("cache_" + std::string(kv) + "_l" + std::to_string(il + m.off));
            m.kv_llama = m.kv_llama && it != t.end() && it->second->type == GGML_TYPE_F16 &&
                         ggml_backend_buffer_is_axera(it->second->buffer) && ggml_is_contiguous(it->second) &&
                         it->second->ne[3] == 1 && (m.N == 0 || it->second->ne[1] == m.N);
            m.N = m.kv_llama ? it->second->ne[1] : m.N;
        }
    }
    // a KV cache no fused attention kernel holds (or GGML_AXERA_FLASH=1): split attention over S-cell segments
    const int64_t ctx = m.kv_llama ? axera_kernels().ctx_for(attn_name + "_kvc", m.N) : 0;
    m.fd.clear();
    m.S     = axera_flash_seg(m.N);
    m.flash = m.kv_llama && m.S > 0 && (ctx == 0 || axera_flash_forced()) && axera_q35_load_flash(m, 1);
    if (m.kv_llama && ctx == 0 && !m.flash) {
        GGML_LOG_WARN("%s: no %s_kvc_c<LMAX> kernel holds the %" PRId64 "-cell KV cache and no split attention "
                      "kernels qwen35_attn_fd_*; lower -c\n", __func__, attn_name.c_str(), m.N);
    }
    m.kv_llama = m.kv_llama && (ctx > 0 || m.flash);
    m.ctx_sfx  = m.kv_llama && !m.flash ? "_kvc_c" + std::to_string(ctx) : "";
    if (m.kv_llama && !m.flash) {
        // fused kernels; split attention over one segment of the whole tier where its kernels exist (long batches)
        m.fd.clear();
        m.S = m.S > 0 && ctx % m.S == 0 ? std::min<int64_t>(ctx, m.S) : ctx;  // the split kernels' segment
        axera_q35_load_flash(m, 1);
    }
    m.gdn  = m.hybrid ? axera_kernels().get(kp + (mx8 ? "_gdn_mx8" : "_gdn")) : nullptr;
    m.attn = m.flash ? nullptr : axera_kernels().get(attn_name + m.ctx_sfx);
    m.attn_all = m.attn && axera_cores() > 1 &&
                 axera_kernels().file(attn_name + m.ctx_sfx) == attn_name + m.ctx_sfx + "_npu" + std::to_string(axera_cores());
    if (m.flash) {  // the KQ mask is cut per segment kernel (fd_kernels::bc); whole 64-cell blocks here
        m.LMAX = (int) m.S;
        m.BC   = 64;
    } else {
        axera_q35_context(m.attn, m.LMAX, m.BC);
    }
    if (m.kv_llama && !m.flash) {
        GGML_ASSERT(m.LMAX == ctx && m.N % m.BC == 0);
    } else if (!m.kv_llama) {
        m.N = m.LMAX;
    }
    m.lm   = lm;
    m.lm_q.reset();
    m.lm_s.reset();
    m.lm_bf.reset();
    m.bf16_of.clear();
    if (!lm) {
        // pooled embeddings: no output projection
    } else if (mx8) {
        // one MXFP8 copy per output projection tensor, shared with the MTP draft context of the same model
        static std::mutex mu;
        static std::map<const ggml_tensor *, std::pair<std::weak_ptr<ggml_backend_axera_buffer_context>,
                                                       std::weak_ptr<ggml_backend_axera_buffer_context>>> shared;
        std::lock_guard<std::mutex> lock(mu);
        // MXFP4 with its mul_mat_mx4_n<N>_k<K> kernel: by default for MXFP4 weights (GGML_AXERA_WEIGHTS=mxfp4, or the MoE
        // models, whose experts are MXFP4), else GGML_AXERA_LMHEAD=mxfp4; GGML_AXERA_LMHEAD=mxfp8 keeps MXFP8
        const char * lh  = std::getenv("GGML_AXERA_LMHEAD");
        const char * wf  = std::getenv("GGML_AXERA_WEIGHTS");
        const bool   lm4 = (lh ? strcmp(lh, "mxfp4") == 0 : m.moe || (wf && strcmp(wf, "mxfp4") == 0)) &&
                         axera_kernels().has("mul_mat_mx4_n" + std::to_string(lm->ne[1]) + "_k" + std::to_string(lm->ne[0]));
        auto & sh = shared[lm4 ? (const ggml_tensor *) ((const char *) lm + 1) : lm];  // a key per format
        m.lm_q    = sh.first.lock();
        m.lm_s    = sh.second.lock();
        if (!m.lm_q || !m.lm_s) {
            axera_cmm q, sc;
            if (lm4) {
                axera_quantize_mxfp4(lm, q, sc);
            } else {
                axera_quantize_mxfp8(lm, q, sc);
            }
            m.lm_q = std::move(q);
            m.lm_s = std::move(sc);
            sh     = { m.lm_q, m.lm_s };
        }
    } else if (lm->type != GGML_TYPE_BF16 || !ggml_backend_buffer_is_axera(lm->buffer)) {  // or in the GGUF mapping
        m.lm_bf = axera_to_bf16(lm);
    }
    auto size_of = axera_input_size;
    GGML_ASSERT(!m.gdn || size_of(m.gdn, "x") == (uint32_t) H * sizeof(ggml_bf16_t));
    ggml_axera_kernel * rope_k = m.flash ? m.fd.at(1).head : m.attn;  // decode cos / sin [1, ROT / 2]
    m.A    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    m.B    = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
    m.NBLK = axera_cmm_alloc(sizeof(int32_t));
    m.COS  = axera_cmm_alloc(size_of(rope_k, "cos"));
    m.SIN  = axera_cmm_alloc(size_of(rope_k, "sin"));
    m.POS.reset();
    m.MASK.reset();
    if (!m.flash) {
        m.POS  = axera_cmm_alloc(size_of(m.attn, "pos"));
        m.MASK = axera_cmm_alloc(size_of(m.attn, "mask"));
        GGML_ASSERT(m.MASK->size >= (size_t) m.LMAX * sizeof(ggml_bf16_t));
    }
    if (size_of(rope_k, "cos") != (uint32_t) (m.ROT / 2) * sizeof(ggml_bf16_t)) {
        GGML_ABORT("qwen35 template: the graph rotates %d dims, the attention kernel %zu\n", m.ROT,
                   2 * size_of(rope_k, "cos") / sizeof(ggml_bf16_t));
    }
    m.inv_freq.resize(m.ROT / 2);
    for (int i = 0; i < m.ROT / 2; ++i) {
        m.inv_freq[i] = (float) (1.0 / std::pow(m.THETA, (double) (2 * i) / m.ROT));
    }
    if (m.moe) {  // the routed experts kernel and its IO: the layer kernels write BASE / H2 / LOGITS
        auto topk = t.find("ffn_moe_topk-" + std::to_string(m.off));
        GGML_ASSERT(topk != t.end() && "qwen35 template: no ffn_moe_topk node in the MoE graph");
        m.top_k    = (int) topk->second->ne[0];
        m.n_expert = (int) t.at("blk." + std::to_string(m.off) + ".ffn_gate_inp.weight")->ne[1];
        m.BASE     = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        m.H2       = axera_cmm_alloc(H * sizeof(ggml_bf16_t));
        m.LOGITS   = axera_cmm_alloc(m.n_expert * sizeof(float));
        m.RW       = axera_cmm_alloc(m.top_k * sizeof(float));
        for (int gu = 0; gu < 2; ++gu) {
            const std::string name = "qwen35moe_experts" + std::to_string(m.top_k) + (gu ? "_gu" : "") + "_mxfp4";
            auto &            ek   = m.ex[gu];
            ek                     = {};
            if (gu && !std::ifstream(axera_kernels().path(name)).good()) {
                continue;  // optional
            }
            ek.k = axera_kernels().get(name);
            GGML_ASSERT(size_of(ek.k, "rw") == m.top_k * sizeof(float) && size_of(ek.k, "h2") == H * sizeof(ggml_bf16_t));
            ek.in.assign(ek.k->info->nInputSize, {});
            for (int g = 0; g < 4; ++g) {
                ek.w[g].assign(m.top_k, -1);
                ek.s[g].assign(m.top_k, -1);
            }
            for (uint32_t i = 0; i < ek.k->info->nInputSize; ++i) {
                const std::string n  = ek.k->info->pInputs[i].pName;
                const uint32_t    sz = ek.k->info->pInputs[i].nSize;
                if (n == "h2" || n == "base" || n == "rw") {
                    auto & c = n == "h2" ? m.H2 : n == "base" ? m.BASE : m.RW;
                    ek.in[i] = axera_io(c.get(), c->vir, sz);
                    continue;
                }
                const char * kinds = "gudc";  // w_<kind><expert> / s_<kind><expert>
                const char * kc    = n.size() > 3 ? strchr(kinds, n[2]) : nullptr;
                const int    e     = kc ? atoi(n.c_str() + 3) : -1;
                if ((n[0] != 'w' && n[0] != 's') || n[1] != '_' || !kc || e < 0 || e >= m.top_k) {
                    GGML_ABORT("qwen35 template: unexpected experts kernel input %s\n", n.c_str());
                }
                (n[0] == 'w' ? ek.w : ek.s)[kc - kinds][e] = (int) i;
            }
            for (int par = 0; par < 2; ++par) {  // even layers write m.B, odd ones m.A
                auto & x = par == 0 ? m.B : m.A;
                ek.out[par].assign(ek.k->info->nOutputSize, {});
                ek.out[par][ek.k->output("out")] = axera_io(x.get(), x->vir, H * sizeof(ggml_bf16_t));
            }
        }
        for (int a = 0; a < 2; ++a) {  // experts + next layer: inputs and outputs mapped by name
            auto &              mk   = m.mx[a];
            ggml_axera_kernel * body = a == 0 ? m.gdn : m.attn;
            const std::string   name = "qwen35moe_experts" + std::to_string(m.top_k) + "_gu_" + (a == 0 ? "gdn" : "attn") +
                                     (mx8 ? "_mx8" : "") + (a == 0 ? "" : m.ctx_sfx);
            mk = {};
            if (!m.ex[1].k || !body || !axera_kernels().has(name)) {
                continue;
            }
            mk.k = axera_kernels().get(name);
            for (uint32_t i = 0; i < mk.k->info->nInputSize; ++i) {
                const char * n = mk.k->info->pInputs[i].pName;
                mk.body.push_back(body->input(n));
                mk.ein.push_back(mk.body.back() < 0 ? m.ex[1].k->input(n) : -1);
                GGML_ASSERT(mk.body.back() >= 0 || mk.ein.back() >= 0);
            }
            for (uint32_t i = 0; i < mk.k->info->nOutputSize; ++i) {
                mk.bout.push_back(body->output(mk.k->info->pOutputs[i].pName));
                GGML_ASSERT(mk.bout.back() >= 0);
            }
            mk.in.resize(mk.body.size());
            mk.out.resize(mk.bout.size());
        }
        for (int a = 0; a < 2; ++a) {  // whole layers: the layer kernel's IO by name, the expert planes, the packs
            auto &              fk   = m.full[a];
            ggml_axera_kernel * body = a == 0 ? m.gdn : m.attn;
            const std::string   base = std::string("qwen35moe_layer_") + (a == 0 ? "gdn" : "attn") + (mx8 ? "_mx8" : "");
            const std::string   sfx  = a == 0 ? "" : m.ctx_sfx;
            std::string         name = base + "_pkc" + sfx;  // the packed build with column blocks first, then packed
            fk = {};
            if (!axera_kernels().has(name)) {
                name = base + "_pk" + sfx;
            }
            if (!axera_kernels().has(name)) {
                name = base + sfx;
            }
            if (!m.ex[1].k || !body || !axera_kernels().has(name)) {
                continue;
            }
            fk.k = axera_kernels().get(name);
            static const char * packs[4] = { "pw", "ps", "pf", "pb" };
            if (name.find("_pk") != std::string::npos) {
                // members: "<member> <pack> <offset> <bytes> [<rows> <row bytes> <a:b>,<a:b>...]" (column blocks)
                std::string   lp = axera_kernels().path(name);
                std::ifstream f(lp.substr(0, lp.size() - strlen(".axmodel")) + ".layout");
                std::string   line;
                GGML_ASSERT(f.good());
                while (std::getline(f, line)) {
                    std::istringstream ls(line);
                    std::string        mn, pk, bl;
                    size_t             off = 0, bytes = 0, rows = 0, row_bytes = 0;
                    if (!(ls >> mn >> pk >> off >> bytes)) {
                        continue;
                    }
                    ls >> rows >> row_bytes >> bl;
                    const int pi = (int) (std::find(packs, packs + 4, mn == "pw" || mn == "ps" || mn == "pf" || mn == "pb" ? mn : pk) - packs);
                    GGML_ASSERT(pi < 4);
                    if (pk == "-") {
                        fk.pack_bytes[pi] = bytes;
                        continue;
                    }
                    const int j = body->input(mn.c_str());
                    if (j < 0) {
                        GGML_ABORT("qwen35 template: %s packs %s, no input of the layer kernel\n", name.c_str(), mn.c_str());
                    }
                    axera_q35::full_kernel::member mm = { pi, j, off, bytes };
                    mm.rows      = rows;
                    mm.row_bytes = row_bytes;
                    for (size_t a0 = 0; a0 < bl.size();) {
                        size_t x = 0, y = 0;
                        GGML_ASSERT(sscanf(bl.c_str() + a0, "%zu:%zu", &x, &y) == 2);
                        mm.blocks.push_back({ x, y });
                        const size_t c = bl.find(',', a0);
                        a0             = c == std::string::npos ? bl.size() : c + 1;
                    }
                    fk.members.push_back(std::move(mm));
                }
            }
            for (uint32_t i = 0; i < fk.k->info->nInputSize; ++i) {
                const std::string n  = fk.k->info->pInputs[i].pName;
                const int         pl = n == "e_gw" ? 0 : n == "e_gs" ? 1 : n == "e_dw" ? 2 : n == "e_ds" ? 3 : -1;
                const int         pi = (int) (std::find(packs, packs + 4, n) - packs);
                fk.plane.push_back(pl);
                fk.pack.push_back(pi < 4 ? pi : -1);
                fk.body.push_back(pl < 0 && pi == 4 ? body->input(n.c_str()) : -1);
                GGML_ASSERT(pl >= 0 || pi < 4 || fk.body.back() >= 0);
                GGML_ASSERT(pi == 4 || fk.k->info->pInputs[i].nSize == fk.pack_bytes[pi]);
            }
            for (uint32_t i = 0; i < fk.k->info->nOutputSize; ++i) {
                const std::string n = fk.k->info->pOutputs[i].pName;
                fk.bout.push_back(n == "out" ? -1 : body->output(n.c_str()));
                GGML_ASSERT(n == "out" || fk.bout.back() >= 0);
            }
            fk.in.resize(fk.body.size());
            fk.out.resize(fk.bout.size());
        }
        m.slots = {};
        std::string sn;  // the slots kernel with the most rows per slot (fewest slots: ~25 us per projection), its
                         // bf16-output build (_bo) first; with planar experts the planar build (_pl, seven IO), most
                         // slots per call first
        m.planar = m.full[0].k || m.full[1].k;
        for (const char * sl : { "128", "64" }) {  // the most slots per call first (a fixed cost per call)
            for (const char * o : { "_bo", "" }) {
                const std::string n = std::string("qwen35moe_slots") + sl + "_g64_px_mxfp4" + o;
                if (sn.empty() && m.ex[1].k && axera_kernels().has(n)) {
                    sn       = n;
                    m.planar = true;
                }
            }
        }
        for (const char * sl : { "64", "32" }) {
            for (const char * g : { "64", "32" }) {
                for (const char * o : { "_bo", "" }) {
                    const std::string n = std::string("qwen35moe_slots") + sl + "_g" + g + "_pl_mxfp4" + o;
                    if (sn.empty() && m.ex[1].k && axera_kernels().has(n)) {
                        sn       = n;
                        m.planar = true;
                    }
                }
            }
        }
        for (const char * g : { "64", "32", "16", "8" }) {
            for (const char * o : { "_bo", "" }) {
                const std::string n = std::string("qwen35moe_slots32_g") + g + "_gu_mxfp4" + o;
                if (sn.empty() && axera_kernels().has(n)) {
                    sn = n;
                }
            }
        }
        if (const char * e = std::getenv("GGML_AXERA_MOE_SLOTS")) {  // e.g. qwen35moe_slots32_g8_gu_mxfp4
            sn = e;
        }
        // a slots kernel's IO indices by name, its slot count S and rows per slot G (from rw [S, G])
        auto setup = [&](axera_q35::slots_kernel & sk, const std::string & sn) {
            sk.k      = axera_kernels().get(sn);
            sk.x      = sk.k->input("x");
            sk.tok    = sk.k->input("tok");
            sk.h2     = sk.k->input("h2");
            sk.rw     = sk.k->input("rw");
            sk.out    = sk.k->output("out");
            GGML_ASSERT((sk.x >= 0 || sk.tok >= 0) && sk.rw >= 0 && sk.out >= 0);
            const AX_ENGINE_IOMETA_T & rm = sk.k->info->pInputs[sk.rw];  // [S, G]
            GGML_ASSERT(rm.nShapeSize == 2);
            sk.S = rm.pShape[0];
            sk.G = rm.pShape[1];
            GGML_ASSERT(sk.x < 0 || size_of(sk.k, "x") == (uint32_t) (sk.S * sk.G * H) * sizeof(ggml_bf16_t));
            sk.ob = axera_output_size(sk.k, "out") == (size_t) sk.S * sk.G * H * sizeof(ggml_bf16_t);
            sk.in.assign(sk.k->info->nInputSize, {});
            sk.outs.assign(sk.k->info->nOutputSize, {});
            for (int g = 0; g < 2; ++g) {
                sk.w[g].assign(sk.S, -1);
                sk.s[g].assign(sk.S, -1);
            }
            for (uint32_t i = 0; i < sk.k->info->nInputSize; ++i) {
                const std::string n = sk.k->info->pInputs[i].pName;
                if (n == "x" || n == "rw" || n == "tok" || n == "h2") {
                    continue;
                }
                const int pl = n == "e_gw" ? 0 : n == "e_gs" ? 1 : n == "e_dw" ? 2 : n == "e_ds" ? 3 : -1;
                if (pl >= 0 || n == "sid") {
                    (pl >= 0 ? sk.plane[pl] : sk.sid) = (int) i;
                    continue;
                }
                const int e = n.size() > 3 ? atoi(n.c_str() + 3) : -1;
                if ((n[0] != 'w' && n[0] != 's') || n[1] != '_' || (n[2] != 'c' && n[2] != 'd') || e < 0 || e >= sk.S) {
                    GGML_ABORT("qwen35 template: unexpected slots kernel input %s\n", n.c_str());
                }
                (n[0] == 'w' ? sk.w : sk.s)[n[2] == 'd'][e] = (int) i;
            }
            if (sk.sid >= 0) {
                sk.SID          = axera_cmm_alloc(sk.S * sizeof(int32_t));
                sk.in[sk.sid]   = axera_io(sk.SID.get(), sk.SID->vir, (uint32_t) (sk.S * sizeof(int32_t)));
            }
        };
        m.slots_px.clear();
        if (m.ex[1].k && !sn.empty()) {
            setup(m.slots, sn);
            const char * e = std::getenv("GGML_AXERA_SLOTS_G");  // 0: the main slots kernel only
            for (int g : { 16, 32 }) {
                const std::string n = "qwen35moe_slots" + std::to_string(m.slots.S) + "_g" + std::to_string(g) + "_px_mxfp4" +
                                      (m.slots.ob ? "_bo" : "");
                if (m.slots.tok >= 0 && g < m.slots.G && !(e && strcmp(e, "0") == 0) && axera_kernels().has(n)) {
                    m.slots_px.emplace_back();
                    setup(m.slots_px.back(), n);
                    GGML_ASSERT(m.slots_px.back().tok >= 0 && m.slots_px.back().ob == m.slots.ob);
                }
            }
            const int gmin = m.slots_px.empty() ? m.slots.G : m.slots_px[0].G;
            if (m.slots.tok >= 0) {  // per slot: expert, G token rows (any chunk; a group's up to its slot rows)
                m.TOK = axera_cmm_alloc((size_t) 4096 * (1 + m.slots.G) * sizeof(int32_t) * (m.slots.G / gmin));
            }
        }
    }
    m.layers.clear();
    m.layers.resize(L);
    m.rollback = true;
    for (int il = 0; il < L; ++il) {
        axera_q35_bind_layer(m, il, t);
        if (m.layers[il].gdn && !m.layers[il].in_place) {
            m.rollback = false;
        }
    }
    // each layer's routed experts, repacked for the kernels: gate-up when the gate-up experts kernel exists (a GGUF
    // ffn_gate_up_exps, gate and up in place when back to back, else into a buffer of their own), else apart
    static std::set<const void *> gu_done;  // gate tensors repacked in place
    m.gu_own.clear();
    for (int il = 0; il < L && m.moe; ++il) {
        auto &              ly = m.layers[il];
        const std::string   b  = "blk." + std::to_string(il + m.off) + ".";
        const auto          gu = t.find(b + "ffn_gate_up_exps.weight");
        const ggml_tensor * w[3];
        w[0] = gu != t.end() ? gu->second : t.at(b + "ffn_gate_exps.weight");
        w[1] = gu != t.end() ? nullptr : t.at(b + "ffn_up_exps.weight");
        w[2] = t.at(b + "ffn_down_exps.weight");
        axera_q35_experts_to_cmm(w);
        for (const ggml_tensor * x : w) {
            if (x && (x->type != GGML_TYPE_MXFP4 || !ggml_backend_buffer_is_axera(x->buffer))) {
                GGML_ABORT("qwen35 template: %s is %s; the MoE kernels take MXFP4 experts in CMM (llama-quantize "
                           "MXFP4_MOE, -ngl 99)\n", x->name, ggml_type_name(x->type));
            }
        }
        if (gu != t.end() && !m.ex[1].k) {
            GGML_ABORT("qwen35 template: %s needs the kernel qwen35moe_experts%d_gu_mxfp4\n", w[0]->name, m.top_k);
        }
        ly.exps_gu = m.ex[1].k != nullptr;
        // the gate-up weight: buffer, start, bytes per expert
        ggml_backend_axera_buffer_context * gb = (ggml_backend_axera_buffer_context *) w[0]->buffer->context;
        uint8_t *                           gp = (uint8_t *) w[0]->data;
        if (gu != t.end()) {
            axera_repack_mxfp4(w[0]);
        } else if (ly.exps_gu && axera_mxfp4_gu_adjacent(w[0], w[1])) {
            if (gu_done.insert(w[0]->data).second) {
                axera_repack_mxfp4_gu(w[0], w[1], gp, gb);
            }
        } else if (ly.exps_gu) {
            m.gu_own.push_back(axera_cmm_alloc(2 * ggml_nbytes(w[0])));
            gb = m.gu_own.back().get();
            gp = (uint8_t *) gb->vir;
            axera_repack_mxfp4_gu(w[0], w[1], gp, gb);
        } else {
            axera_repack_mxfp4(w[0]);
            axera_repack_mxfp4(w[1]);
        }
        axera_repack_mxfp4(w[2]);
        const auto & ek = m.ex[ly.exps_gu];
        for (int g = 0; g < 3; ++g) {
            auto & e = ly.exps[g];
            e        = {};
            if (ly.exps_gu && g == 1) {
                continue;
            }
            const bool    two  = ly.exps_gu && g == 0 && gu == t.end();  // gate and up rows of two GGUF tensors
            const int64_t rows = w[g]->ne[1] * (two ? 2 : 1);
            e.b                = g == 0 ? gb : (ggml_backend_axera_buffer_context *) w[g]->buffer->context;
            e.base             = g == 0 ? (char *) gp : (char *) w[g]->data;
            e.data             = (size_t) (w[g]->ne[0] * rows / 2);
            e.scale            = (size_t) (w[g]->ne[0] / 32 * rows);
            e.stride           = e.data + e.scale;
            const int kind     = ly.exps_gu && g == 0 ? 3 : g;
            GGML_ASSERT(e.stride == (two ? 2 : 1) * w[g]->nb[2] && e.data == ek.k->info->pInputs[ek.w[kind][0]].nSize &&
                        e.scale == ek.k->info->pInputs[ek.s[kind][0]].nSize);
            e.n = w[g]->ne[2];
        }
        if (ly.exps_gu && m.planar) {  // whole layers and the planar slots kernel take the experts as planes
            for (int g : { 0, 2 }) {
                auto & e = ly.exps[g];
                axera_mxfp4_planar(e.b, e.base, e.n, e.data, e.scale);
                e.planar = true;
            }
        }
    }
    if (!m.gu_own.empty()) {
        GGML_LOG_INFO("%s: %zu MoE layers with gate and up experts apart, repacked into %zu MB of their own\n", __func__,
                      m.gu_own.size(), m.gu_own.size() * m.gu_own[0]->size >> 20);
    }
    // prefill kernels (every chunk size found, largest first). With MXFP8 decode, prefill takes MXFP8 weights too when
    // the kernel directory has MXFP8 prefill kernels (built where prefill is bound by reading the weights, 9B: about
    // half the cycles), else the bf16 GGUF weights (0.8B: compute bound, MXFP8 is slower);
    // GGML_AXERA_PREFILL_WEIGHTS=mxfp8 / bf16 chooses
    const char * pw     = std::getenv("GGML_AXERA_PREFILL_WEIGHTS");
    bool         pre_mx = false;
    for (int P : { 128, 64, 32, 16 }) {
        const std::string p = std::to_string(P);
        pre_mx = pre_mx || std::ifstream(axera_kernels().path(m.hybrid ? "qwen35_gdn_prefill" + p + "_mx8" :
                                                                         m.ap + "_attn_fd_head" + p + "_mx8")).good() ||
                 (!m.hybrid && std::ifstream(axera_kernels().path(m.ap + "_attn_prefill" + p + "_mx8" + m.ctx_sfx)).good());
    }
    pre_mx = mx8 && (pw != nullptr ? strcmp(pw, "mxfp8") == 0 : pre_mx);
    m.pre_mx = pre_mx;
    m.pre.clear();
    for (int P : { 256, 128, 64, 32, 16, 8, 4, 3, 2, 1 }) {
        if (m.moe) {  // MoE: qwen35moe_{gdn,attn}_prefill<P> (bf16 weights) with the slots kernel
            // attention: the fused kernel of the context tier; split (flash) only when no fused kernel holds the
            // context (m.flash), never in place of a missing chunk size of a fused tier
            // the GDN layer in MXFP8 where that kernel exists (prefill reads the weights: about 20% faster)
            std::string gn = "qwen35moe_gdn_prefill" + std::to_string(P);
            gn += mx8 && std::ifstream(axera_kernels().path(gn + "_mx8")).good() ? "_mx8" : "";
            // the builds that also output the router's top-K (_tk) first (GGML_AXERA_PREFILL_TK=0: not)
            const char * tke = std::getenv("GGML_AXERA_PREFILL_TK");
            const bool   tk  = !(tke && strcmp(tke, "0") == 0);
            gn += tk && axera_kernels().has(gn + "_tk") ? "_tk" : "";
            std::string an = "qwen35moe_attn_prefill" + std::to_string(P) + "_tk" + m.ctx_sfx;
            if (!tk || !std::ifstream(axera_kernels().path(an)).good()) {
                an = "qwen35moe_attn_prefill" + std::to_string(P) + m.ctx_sfx;
            }
            const bool fused = !m.flash && std::ifstream(axera_kernels().path(an)).good();
            const bool split = m.kv_llama && (m.flash || (!m.attn_all && !fused)) && axera_q35_load_flash(m, P);
            if (P == 1 || !m.slots.k || !std::ifstream(axera_kernels().path(gn)).good() || !(fused || split)) {
                continue;
            }
            axera_q35::prefill pf;
            pf.P    = P;
            pf.gdn  = axera_kernels().get(gn);
            pf.attn = fused ? axera_kernels().get(an) : nullptr;
            axera_q35_load_pk(pf.gdn_pk, gn, pf.gdn);
            axera_q35_load_pk(pf.attn_pk, an, pf.attn);
            GGML_ASSERT(size_of(pf.gdn, "x") == (uint32_t) P * H * sizeof(ggml_bf16_t));
            if (pf.attn) {
                pf.R = (int64_t) (size_of(pf.attn, "mask") / sizeof(ggml_bf16_t)) / m.LMAX;
                GGML_ASSERT(pf.R == P && (size_t) pf.R * m.LMAX * sizeof(ggml_bf16_t) == size_of(pf.attn, "mask"));
                pf.MASK = axera_cmm_alloc(size_of(pf.attn, "mask"));
            }
            ggml_axera_kernel * rk = pf.attn ? pf.attn : m.fd.at(P).head;
            pf.A    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.B    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.COS  = axera_cmm_alloc(size_of(rk, "cos"));
            pf.SIN  = axera_cmm_alloc(size_of(rk, "sin"));
            pf.BASE = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.H2   = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
            pf.LOG  = axera_cmm_alloc((size_t) P * m.n_expert * sizeof(float));
            pf.gdn_tk  = pf.gdn && pf.gdn->output("topk_i") >= 0;
            pf.attn_tk = pf.attn && pf.attn->output("topk_i") >= 0;
            if (pf.gdn_tk || pf.attn_tk) {
                pf.TKV = axera_cmm_alloc((size_t) P * m.top_k * sizeof(float));
                pf.TKI = axera_cmm_alloc((size_t) P * m.top_k * sizeof(int32_t));
            }
            // at most one partly filled slot per expert
            const int    G     = m.slots.G;
            const size_t slots = (size_t) GGML_PAD(std::min(P * m.top_k, m.n_expert + (P * m.top_k + G - 1) / G), m.slots.S);
            const size_t rows  = slots * G;
            if (!m.SX || m.SX->size < rows * H * sizeof(ggml_bf16_t)) {
                m.SX   = axera_cmm_alloc_nc(rows * H * sizeof(ggml_bf16_t));
                m.SRW  = axera_cmm_alloc(rows * sizeof(float));
                m.SOUT = axera_cmm_alloc(rows * H * sizeof(float));
            }
            m.pre.push_back(std::move(pf));
            continue;
        }
        const std::string sfx  = std::to_string(P) + (pre_mx ? "_mx8" : "");
        const std::string asfx = sfx + m.ctx_sfx;
        const bool one = (!m.hybrid || std::ifstream(axera_kernels().path("qwen35_gdn_prefill" + sfx)).good()) &&
                         (m.flash ? axera_q35_load_flash(m, P) :
                                    std::ifstream(axera_kernels().path(m.ap + "_attn_prefill" + asfx)).good());
        if (!one) {
            continue;
        }
        if (m.kv_llama && !m.flash) {
            axera_q35_load_flash(m, P);  // optional
        }
        axera_q35::prefill pf;
        pf.P    = P;
        pf.gdn  = m.hybrid ? axera_kernels().get("qwen35_gdn_prefill" + sfx) : nullptr;
        if (pf.gdn && std::ifstream(axera_kernels().path("qwen35_gdn_prefill" + sfx + "_rs")).good()) {
            pf.gdn_rs = axera_kernels().get("qwen35_gdn_prefill" + sfx + "_rs");
        }
        pf.attn = !m.flash ? axera_kernels().get(m.ap + "_attn_prefill" + asfx) : nullptr;
        GGML_ASSERT(!pf.gdn || size_of(pf.gdn, "x") == (uint32_t) P * H * sizeof(ggml_bf16_t));
        ggml_axera_kernel * rk = m.flash ? m.fd.at(P).head : pf.attn;
        if (pf.attn) {
            // chunk mask [LMAX / BC][R][BC] (R = P rows, or G * P in older kernels)
            pf.R = (int64_t) (size_of(pf.attn, "mask") / sizeof(ggml_bf16_t)) / m.LMAX;
            GGML_ASSERT(pf.R % P == 0 && (size_t) pf.R * m.LMAX * sizeof(ggml_bf16_t) == size_of(pf.attn, "mask"));
            pf.MASK = axera_cmm_alloc(size_of(pf.attn, "mask"));
        }
        pf.A    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
        pf.B    = axera_cmm_alloc(P * H * sizeof(ggml_bf16_t));
        pf.COS  = axera_cmm_alloc(size_of(rk, "cos"));
        pf.SIN  = axera_cmm_alloc(size_of(rk, "sin"));
        m.pre.push_back(std::move(pf));
    }
    // batched decode kernels (P = B tokens of B sequences, largest first): with the recurrent state and the KV cache
    // in the llama.cpp memory. In the decode weight format (_mx8); bf16 kernels next to MXFP8 decode kernels use the
    // bf16 GGUF tensors
    for (int Bn : { 8, 4, 2 }) {
        const std::string b  = std::to_string(Bn);
        const std::string bs = mx8 && std::ifstream(axera_kernels().path(m.ap + "_attn_batch" + b + "_mx8" + m.ctx_sfx)).good() ? "_mx8" : "";
        const std::string gn = "qwen35_gdn_batch" + b + bs, an = m.ap + "_attn_batch" + b + bs;
        if (m.moe || !m.kv_llama || !m.rollback || (m.hybrid && !std::ifstream(axera_kernels().path(gn)).good()) ||
            !(m.flash ? axera_q35_load_flash(m, Bn) : std::ifstream(axera_kernels().path(an + m.ctx_sfx)).good())) {
            continue;
        }
        axera_q35::prefill pf;
        pf.P     = Bn;
        pf.batch = true;
        pf.gdn   = m.hybrid ? axera_kernels().get(gn) : nullptr;
        pf.attn  = m.flash ? nullptr : axera_kernels().get(an + m.ctx_sfx);
        GGML_ASSERT(!pf.gdn || size_of(pf.gdn, "x") == (uint32_t) Bn * H * sizeof(ggml_bf16_t));
        ggml_axera_kernel * rk = m.flash ? m.fd.at(Bn).head : pf.attn;
        if (!m.flash) {
            pf.R = (int64_t) (size_of(pf.attn, "mask") / sizeof(ggml_bf16_t)) / m.LMAX;
            GGML_ASSERT(pf.R == Bn);
            pf.MASK = axera_cmm_alloc(size_of(pf.attn, "mask"));
            pf.POS  = axera_cmm_alloc(size_of(pf.attn, "pos"));
        }
        pf.A    = axera_cmm_alloc(Bn * H * sizeof(ggml_bf16_t));
        pf.B    = axera_cmm_alloc(Bn * H * sizeof(ggml_bf16_t));
        pf.COS  = axera_cmm_alloc(size_of(rk, "cos"));
        pf.SIN  = axera_cmm_alloc(size_of(rk, "sin"));
        m.pre.push_back(std::move(pf));
    }
    // MoE groups: copies of the largest chunk, GGML_AXERA_MOE_GROUP chunks per group (default: 2048 tokens, a whole
    // ubatch of -ub 2048; the experts read their weights once per group). With fused attention kernels each copy has
    // its own mask, pos and nblk inputs (the chunks of a group run interleaved, layer by layer)
    if (m.moe && !m.pre.empty() && m.pre[0].P > 1 && !m.pre[0].batch) {
        const char * ge = std::getenv("GGML_AXERA_MOE_GROUP");
        const int    P  = m.pre[0].P;
        const int    Q  = std::max(1, ge ? atoi(ge) : 2048 / P);
        m.pre[0].group  = { 0 };
        for (int c = 1; c < Q; ++c) {
            axera_q35::prefill pc;
            pc.P    = P;
            pc.gdn  = m.pre[0].gdn;
            pc.gdn_pk = m.pre[0].gdn_pk;
            pc.sub  = c;
            pc.fdk  = P + 1000 * c;
            pc.A    = axera_cmm_alloc(m.pre[0].A->size);
            pc.B    = axera_cmm_alloc(m.pre[0].B->size);
            pc.COS  = axera_cmm_alloc(m.pre[0].COS->size);
            pc.SIN  = axera_cmm_alloc(m.pre[0].SIN->size);
            pc.BASE = axera_cmm_alloc(m.pre[0].BASE->size);
            pc.H2   = axera_cmm_alloc(m.pre[0].H2->size);
            pc.LOG  = axera_cmm_alloc(m.pre[0].LOG->size);
            pc.gdn_tk  = m.pre[0].gdn_tk;
            pc.attn_tk = m.pre[0].attn_tk;
            if (m.pre[0].TKV) {
                pc.TKV = axera_cmm_alloc(m.pre[0].TKV->size);
                pc.TKI = axera_cmm_alloc(m.pre[0].TKI->size);
            }
            if (m.pre[0].attn) {
                pc.attn  = m.pre[0].attn;
                pc.attn_pk = m.pre[0].attn_pk;
                pc.R     = m.pre[0].R;
                pc.MASK  = axera_cmm_alloc(m.pre[0].MASK->size);
                pc.CPOS  = axera_cmm_alloc(m.POS->size);
                pc.CNBLK = axera_cmm_alloc(m.NBLK->size);
            }
            if (m.fd.count(P)) {
                axera_q35_clone_flash(m, P, pc.fdk);
            }
            m.pre[0].group.push_back((int) m.pre.size());
            m.pre.push_back(std::move(pc));
        }
        const int    NT    = P * Q, G = m.slots.G;  // the most slot rows (TOK: sized for the smallest G above)
        m.comb             = {};
        const std::string cn = "qwen35moe_combine" + std::to_string(m.top_k) + "_t" + std::to_string(NT);
        for (const auto & f : { std::string("_r32768") }) {
            if (m.slots.tok >= 0 && axera_kernels().has(cn + f)) {
                m.comb.k = axera_kernels().get(cn + f);
            }
        }
        m.comb_small.clear();
        for (int t = 256; m.comb.k && t < NT; t *= 2) {
            const std::string n = "qwen35moe_combine" + std::to_string(m.top_k) + "_t" + std::to_string(t) + "_r32768";
            if (axera_kernels().has(n)) {
                axera_q35::combine_kernel cs;
                cs.k = axera_kernels().get(n);
                cs.T = t;
                cs.R = (int) (size_of(cs.k, "sout") / (H * sizeof(ggml_bf16_t)));
                GGML_ASSERT(cs.R == (int) (size_of(m.comb.k, "sout") / (H * sizeof(ggml_bf16_t))));
                m.comb_small.push_back(std::move(cs));
            }
        }
        if (m.comb.k) {  // the chunks' rows in one buffer each: views
            m.comb.T = NT;
            m.comb.R = (int) (size_of(m.comb.k, "sout") / (H * sizeof(ggml_bf16_t)));
            const size_t xr = (size_t) P * H * sizeof(ggml_bf16_t), lr = (size_t) P * m.n_expert * sizeof(float);
            m.GA    = axera_cmm_alloc(xr * Q);
            m.GB    = axera_cmm_alloc(xr * Q);
            m.GH2   = axera_cmm_alloc(xr * Q);
            m.GBASE = axera_cmm_alloc(xr * Q);
            m.GLOG  = axera_cmm_alloc(lr * Q);
            const size_t tr = (size_t) P * m.top_k * sizeof(float);  // top-K rows of a chunk (values / experts)
            if (m.pre[0].TKV) {
                m.GTKV = axera_cmm_alloc(tr * Q);
                m.GTKI = axera_cmm_alloc(tr * Q);
            }
            if (m.TOK->size < (size_t) m.comb.R / G * (1 + G) * sizeof(int32_t)) {
                m.TOK = axera_cmm_alloc((size_t) m.comb.R / G * (1 + G) * sizeof(int32_t));
            }
            m.LOC   = axera_cmm_alloc((size_t) NT * m.top_k * sizeof(int32_t));
            for (int c = 0; c < Q; ++c) {
                auto & pc = m.pre[m.pre[0].group[c]];
                pc.A      = axera_cmm_view(m.GA.get(), c * xr, xr);
                pc.B      = axera_cmm_view(m.GB.get(), c * xr, xr);
                pc.H2     = axera_cmm_view(m.GH2.get(), c * xr, xr);
                pc.BASE   = axera_cmm_view(m.GBASE.get(), c * xr, xr);
                pc.LOG    = axera_cmm_view(m.GLOG.get(), c * lr, lr);
                if (m.GTKV) {
                    pc.TKV = axera_cmm_view(m.GTKV.get(), c * tr, tr);
                    pc.TKI = axera_cmm_view(m.GTKI.get(), c * tr, tr);
                }
            }
            GGML_ASSERT(size_of(m.comb.k, "base") == xr * Q && size_of(m.comb.k, "loc") == m.LOC->size);
        }
        const size_t slots = (size_t) GGML_PAD(std::min(NT * m.top_k, m.n_expert + (NT * m.top_k + G - 1) / G), m.slots.S);
        const size_t rows  = slots * G;
        if (m.SX->size < rows * H * sizeof(ggml_bf16_t)) {
            m.SX   = axera_cmm_alloc_nc(rows * H * sizeof(ggml_bf16_t));
            m.SRW  = axera_cmm_alloc(rows * sizeof(float));
            m.SOUT = axera_cmm_alloc(rows * H * sizeof(float));
        }
        if (m.comb.k) {
            GGML_ASSERT(rows <= (size_t) m.comb.R && m.SOUT->size >= size_of(m.comb.k, "sout"));
            if (m.SRW->size < (size_t) m.comb.R * sizeof(float)) {
                m.SRW = axera_cmm_alloc((size_t) m.comb.R * sizeof(float));
            }
        }
    }
    for (size_t v = 0; v < m.pre.size(); ++v) {
        for (int il = 0; il < L; ++il) {
            axera_q35_bind_prefill(m, il, v, t);
        }
    }
    for (int il = 0; il < L && m.kv_llama; ++il) {
        for (int v = -1; v < (int) m.pre.size() && !m.layers[il].gdn; ++v) {
            const int R = v < 0 ? 1 : m.pre[v].P;
            if (m.fd.count(R) && m.fd.at(R).head) {
                axera_q35_bind_flash(m, il, v, t);
            }
        }
    }
    std::string split;
    for (const auto & [R, fk] : m.fd) {
        if (!fk.head) {
            continue;
        }
        split += (split.empty() ? "" : "/") + std::to_string(R) + (fk.cores > 1 ? "x" + std::to_string(fk.cores) : "");
    }
    std::string chunks, batches;
    for (auto & pf : m.pre) {
        if (pf.sub) {
            continue;
        }
        auto & str = pf.batch ? batches : chunks;
        str += (str.empty() ? "" : "/") + std::to_string(pf.P);
    }
    if (!chunks.empty()) {
        chunks += pre_mx ? " (MXFP8)" : " (BF16)";
    }
    m.mgrp = {};
    for (int G : { 8, 4 }) {  // layer groups: the most layers per call whose kernel exists for the layers' kinds
        if (!m.moe || m.mgrp.k || L < G || !m.full[0].k || !m.full[1].k || std::getenv("GGML_AXERA_NO_GROUP")) {
            continue;
        }
        std::string kinds;
        bool        same = true;  // every group of the same kinds
        for (int il = 0; il < L / G * G; ++il) {
            const char c = m.layers[il].gdn ? 'g' : 'a';
            if (il < G) {
                kinds += c;
            } else {
                same = same && kinds[il % G] == c;
            }
        }
        const std::string name = "qwen35moe_group" + std::to_string(G) + "_" + kinds + "_mx8" + m.ctx_sfx;
        if (!same || !mx8 || !axera_kernels().has(name)) {
            continue;
        }
        auto & gk = m.mgrp;
        gk.k      = axera_kernels().get(name);
        gk.G      = G;
        static const char * packs[4] = { "pw", "ps", "pf", "pb" };
        // members: "L<j>.<member> <pack> <offset> <bytes> [<rows> <row bytes> <a:b>,<a:b>...]", the pack sizes "<pack> -"
        std::string   lp = axera_kernels().path(name);
        std::ifstream f(lp.substr(0, lp.size() - strlen(".axmodel")) + ".layout");
        std::string   line;
        GGML_ASSERT(f.good());
        while (std::getline(f, line)) {
            std::istringstream ls(line);
            std::string        mn, pk, bl;
            size_t             off = 0, bytes = 0, rows = 0, row_bytes = 0;
            if (!(ls >> mn >> pk >> off >> bytes)) {
                continue;
            }
            ls >> rows >> row_bytes >> bl;
            const int pi = (int) (std::find(packs, packs + 4, pk == "-" ? mn : pk) - packs);
            GGML_ASSERT(pi < 4);
            if (pk == "-") {
                gk.pack_bytes[pi] = bytes;
                continue;
            }
            const size_t dot = mn.find('.');
            GGML_ASSERT(mn[0] == 'L' && dot != std::string::npos);
            axera_q35::moe_group_kernel::member mm;
            mm.pack      = pi;
            mm.off       = off;
            mm.bytes     = bytes;
            mm.rows      = rows;
            mm.row_bytes = row_bytes;
            mm.j         = atoi(mn.c_str() + 1);
            mm.name      = mn.substr(dot + 1);
            mm.body      = (m.layers[mm.j].gdn ? m.gdn : m.attn)->input(mm.name.c_str());
            if (mm.body < 0 || mm.j >= G) {
                GGML_ABORT("qwen35 template: %s packs %s, no input of the layer kernel\n", name.c_str(), mn.c_str());
            }
            for (size_t a0 = 0; a0 < bl.size();) {
                size_t x = 0, y = 0;
                GGML_ASSERT(sscanf(bl.c_str() + a0, "%zu:%zu", &x, &y) == 2);
                mm.blocks.push_back({ x, y });
                const size_t c = bl.find(',', a0);
                a0             = c == std::string::npos ? bl.size() : c + 1;
            }
            gk.members.push_back(std::move(mm));
        }
        // IO: x / out, the packs, e_<plane><j>, the shared attention inputs (layer 0's names, its first attention
        // layer's buffers), per layer <st|cv|kc|vc><j>: the layer kernel's state_in / conv_in / kc / vc
        static const char * planes[4] = { "e_gw", "e_gs", "e_dw", "e_ds" };
        auto layer_io = [&](const std::string & n, bool out) {
            axera_q35::moe_group_kernel::io io;
            const int pi = (int) (std::find(packs, packs + 4, n) - packs);
            if (n == "x" || n == "out") {
                io.kind = n == "x" ? 0 : 4;
                return io;
            }
            if (pi < 4) {
                io.kind = 1;
                io.idx  = pi;
                return io;
            }
            size_t d = n.size();
            while (d > 0 && isdigit((unsigned char) n[d - 1])) {
                --d;
            }
            const std::string stem = n.substr(0, d);
            io.j                   = d < n.size() ? atoi(n.c_str() + d) : (int) (kinds.find('a'));
            GGML_ASSERT(io.j >= 0 && io.j < G);
            const int pl = (int) (std::find(planes, planes + 4, stem) - planes);
            if (pl < 4) {
                io.kind = 2;
                io.idx  = pl;
                return io;
            }
            io.kind = 3;
            io.n    = stem == "st" ? (out ? "state_out" : "state_in") : stem == "cv" ? (out ? "conv_out" : "conv_in") : stem;
            ggml_axera_kernel * body = m.layers[io.j].gdn ? m.gdn : m.attn;
            io.idx                   = out ? body->output(io.n.c_str()) : body->input(io.n.c_str());
            if (io.idx < 0) {
                GGML_ABORT("qwen35 template: %s: %s %s, not in the layer kernel\n", name.c_str(), out ? "output" : "input",
                           n.c_str());
            }
            return io;
        };
        for (uint32_t i = 0; i < gk.k->info->nInputSize; ++i) {
            gk.ins.push_back(layer_io(gk.k->info->pInputs[i].pName, false));
            GGML_ASSERT(gk.ins.back().kind != 1 || gk.k->info->pInputs[i].nSize == gk.pack_bytes[gk.ins.back().idx]);
        }
        for (uint32_t i = 0; i < gk.k->info->nOutputSize; ++i) {
            gk.outs.push_back(layer_io(gk.k->info->pOutputs[i].pName, true));
        }
        gk.in.resize(gk.ins.size());
        gk.out.resize(gk.outs.size());
        gk.packs.resize(L / G);
        for (int g = 0; g < L / G; ++g) {  // the bound weights of the group's layers into its packs
            for (int pi = 0; pi < 4; ++pi) {
                gk.packs[g][pi] = gk.pack_bytes[pi] ? axera_cmm_alloc(gk.pack_bytes[pi]) : axera_cmm{};
            }
            for (const auto & mm : gk.members) {
                axera_q35_pack_member(m.layers[g * G + mm.j], mm, gk.packs[g][mm.pack], gk.pack_bytes[mm.pack]);
            }
            for (auto & pk : gk.packs[g]) {
                if (pk) {
                    pk->flush(pk->vir, pk->size);
                }
            }
        }
        GGML_LOG_INFO("%s: %s: %d layers per call\n", __func__, name.c_str(), G);
    }
    for (int il = 0; il < L && m.moe; ++il) {  // packed whole layers: the bound weights into the layer's packs
        auto &       ly = m.layers[il];
        const auto & fk = m.full[ly.gdn ? 0 : 1];
        if (!fk.k || fk.members.empty() || (m.mgrp.k && il < L / m.mgrp.G * m.mgrp.G)) {
            continue;
        }
        for (int pi = 0; pi < 4; ++pi) {
            ly.packs[pi] = fk.pack_bytes[pi] ? axera_cmm_alloc(fk.pack_bytes[pi]) : axera_cmm{};
        }
        for (const auto & mm : fk.members) {
            axera_q35_pack_member(ly, mm, ly.packs[mm.pack], fk.pack_bytes[mm.pack]);
        }
        for (int pi = 0; pi < 4; ++pi) {
            if (ly.packs[pi]) {
                ly.packs[pi]->flush(ly.packs[pi]->vir, ly.packs[pi]->size);
            }
        }
    }
    m.ready = true;
    GGML_LOG_INFO("%s: Axera qwen35 template: %d layers, context %" PRId64 " (kernels up to %d), %s weights, recurrent "
                  "state %s, KV cache %s, prefill chunks %s, batched decode %s, split attention rows %s%s\n", __func__, L,
                  m.N, m.LMAX,
                  mx8 ? "MXFP8" : "BF16",
                  m.rollback ? "in the llama.cpp cache (checkpoints work)" : "private",
                  m.flash ? "the llama.cpp cache, attention split over segments (flash decoding)" :
                  m.kv_llama ? "the llama.cpp cache (prompt cache works)" : "private", chunks.empty() ? "none" : chunks.c_str(),
                  batches.empty() ? "none" : batches.c_str(), split.empty() ? "none" : split.c_str(),
                  m.flash || split.empty() ? "" : (" from " + std::to_string(axera_split_min()) + " cells").c_str());
}

// point every delta-net layer's state IO (decode and prefill kernels) at recurrent-cache row `row` (a sequence)
void axera_q35_set_row(axera_q35 & m, int64_t row) {
    for (auto & ly : m.layers) {
        if (ly.row == row) {
            continue;
        }
        for (auto & rg : ly.cache_state) {
            const AX_ENGINE_IO_BUFFER_T io = axera_io(rg.b, rg.base + row * rg.stride, (uint32_t) rg.size);
            ly.in[rg.in]                   = io;
            ly.out[rg.out]                 = io;
            for (auto & pre : ly.pre) {
                if (!pre.batch && !pre.in.empty()) {  // empty: a multi-core-only chunk (its gdn reads ly.in per call)
                    pre.in[rg.conv ? pre.ci : pre.si]  = io;
                    pre.out[rg.conv ? pre.co : pre.so] = io;
                }
            }
        }
        ly.row = row;
    }
}

// point the batched decode IO v of every delta-net layer at the recurrent-cache rows row0 .. row0 + P - 1
void axera_q35_set_batch_rows(axera_q35 & m, size_t v, int64_t row0) {
    for (auto & ly : m.layers) {
        auto & pre = ly.pre[v];
        for (auto & rg : ly.cache_state) {
            const AX_ENGINE_IO_BUFFER_T io =
                axera_io(rg.b, rg.base + row0 * rg.stride, (uint32_t) (m.pre[v].P * rg.stride));
            pre.in[rg.conv ? pre.ci : pre.si]  = io;
            pre.out[rg.conv ? pre.co : pre.so] = io;
        }
    }
}

// clear recurrent-cache row `row` (a new sequence) in every delta-net layer
void axera_q35_zero_row(axera_q35 & m, int64_t row) {
    for (auto & ly : m.layers) {
        for (auto & rg : ly.cache_state) {
            char * p = rg.base + row * rg.stride;
            memset(p, 0, rg.size);
            rg.b->flush(p, rg.size);
        }
    }
}

// point every attention layer's KV IO (decode and prefill kernels) at KV cache stream `s` (llama.cpp's per-sequence
// streams without a unified KV cache)
void axera_q35_set_stream(axera_q35 & m, int64_t s) {
    for (auto & ly : m.layers) {
        if (ly.stream == s) {
            continue;
        }
        for (auto & kr : ly.kv) {
            const AX_ENGINE_IO_BUFFER_T io = axera_io(kr.b, kr.base + s * kr.stride, (uint32_t) kr.size);
            ly.out[kr.out]                 = io;
            for (auto & pre : ly.pre) {
                pre.out[kr.v ? pre.vo : pre.ko] = io;
            }
        }
        ly.stream = s;
    }
}
void axera_q35_restart(axera_q35 & m) {
    for (auto & ly : m.layers) {
        for (auto * s : ly.state) {
            memset(s->vir, 0, s->size);
            s->flush(s->vir, s->size);
        }
    }
    if (m.MASK) {
        auto * mask = (ggml_bf16_t *) m.MASK->vir;
        for (int j = 0; j < m.LMAX; ++j) {
            mask[j] = ggml_fp32_to_bf16(-3e38f);
        }
        m.MASK->flush(m.MASK->vir, m.MASK->size);
    }
    m.next_pos = 0;
}
