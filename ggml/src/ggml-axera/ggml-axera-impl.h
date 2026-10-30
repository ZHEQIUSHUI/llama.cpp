// Axera NPU backend: internal declarations shared by the ggml-axera sources (model independent). The graph
// templates build on them: ggml-axera-attn.h (split attention on the llama.cpp KV cache, RoPE), ggml-axera-q35.h
// (Qwen3.5), the vision encoder in ggml-axera-vit.cpp
#pragma once

// Axera NPU backend.
//
// - Device memory is CMM allocated cached (AX_SYS_MemAllocCached). It is host-visible, so the CPU backend works on
//   it directly for every op this backend does not take; the NPU addresses it physically. Cache maintenance: the CPU
//   side flushes after writing (set_tensor, staging), the NPU side flushes its output range before a run (no dirty
//   line may be evicted over the result) and invalidates it afterwards.
// - Kernels are precompiled axmodels (AXDSL), looked up by name in GGML_AXERA_KERNEL_DIR (default ./axera_kernels).
//   An op is supported only if the kernel for its shape exists; everything else falls back to the CPU backend.
// - Kernel files are <name>_npu<N>.axmodel, N the cores the kernel was compiled for; the code names kernels by <name>.
// - Ops: MUL_MAT with BF16 weights and F32 activations. Kernel "mul_mat_bf16_n<C>_k<K>" (x [1, K] bf16,
//   w [C, K] bf16 -> out [1, C] f32) computes one activation row against C weight rows; N is covered by N / C calls
//   (C = N, or the largest kernel C that divides N, e.g. a vocabulary-sized output projection in chunks). Several
//   activation rows (prompt processing) run row by row, rereading the weights each time, until a batched kernel exists.
// - GGML_AXERA_CORES=N (default 4) runs <name>_npuN (AXDSL kernels split over the N cores without synchronization)
//   where it exists, else <name>_npu1; GGML_AXERA_CORES=1 only the single-core builds. Multi-core kernels need all
//   cores: other NPU users at the same time make the calls time out.
//
// GGML_AXERA_TEMPLATE=qwen35 runs whole Qwen3.5 graphs with per-layer kernels instead (ggml-axera-q35.h).
// GGML_AXERA_DEBUG=1 logs every missing kernel once.

#include "ggml-axera.h"
#include "core/quant.h"
#include "core/runtime.h"
#include "mx.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"

#include <ax_engine_api.h>
#include <ax_sys_api.h>

#include <dirent.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <numeric>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

struct ggml_axera_kernel;
struct ggml_axera_kernel_registry;
struct ggml_backend_axera_context;
struct axera_q35;
struct axera_vit;
struct axera_audio;
struct axera_gen;
namespace axera {
struct model;
}

// functions of the backend sources (ggml-axera-*.cpp)
bool axera_debug();
ggml_axera_kernel_registry & axera_kernels();
std::string axera_mul_mat_kernel_name(int64_t c, int64_t k, bool mx8 = false);
bool ggml_backend_buffer_is_axera(ggml_backend_buffer_t buffer);
ggml_backend_buffer_type_t ggml_backend_axera_mapped_buffer_type(void);
ggml_backend_buffer_t axera_mapped_buffer(void * ptr, size_t size);
bool axera_template_qwen35();
int64_t axera_split_min();
int axera_cores();
bool axera_flash_forced();
int64_t axera_flash_seg(int64_t n);
bool axera_weights_mx8();
bool axera_weights_mx4();
const char * axera_mx_sfx();  // the layer kernels' weight format suffix: _mx8, _mx4 (GGML_AXERA_WEIGHTS=mxfp4)
void axera_gemv_raw(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                    void * wdata, int64_t N, int64_t K, const float * x, ggml_tensor * dst, float * out);
void axera_gemv_raw_to(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx,
                       void * wdata, int64_t N, int64_t K, const float * x,
                       const ggml_backend_axera_buffer_context * dctx, float * out);
bool axera_gemv_batch(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * wctx, void * wdata,
                      const ggml_backend_axera_buffer_context * sctx, int64_t N, int64_t K, const float * x,
                      int Bn, float * const * out);
void axera_gemv(ggml_backend_axera_context * ctx, const ggml_tensor * w, const float * x, ggml_tensor * dst,
                float * out);
void axera_gemv_mx8(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * q,
                    const ggml_backend_axera_buffer_context * sc, int64_t N, int64_t K, const float * x,
                    ggml_tensor * dst, float * out);
void axera_gemv_mx4(ggml_backend_axera_context * ctx, const ggml_backend_axera_buffer_context * q,
                    const ggml_backend_axera_buffer_context * sc, int64_t N, int64_t K, const float * x,
                    ggml_tensor * dst, float * out);
void ggml_backend_axera_mul_mat(ggml_backend_axera_context * ctx, ggml_tensor * dst);
float axera_rms_eps(const ggml_tensor * rms_norm);
uint32_t axera_input_size(ggml_axera_kernel * k, const char * name);
uint32_t axera_output_size(ggml_axera_kernel * k, const char * name);
std::string axera_base_name(const char * name);
void axera_legacy_preload();  // a kernel directory without a manifest: all its kernels
void axera_warn_one_core(ggml_axera_kernel * k);  // the first run of a single-core fallback build: warn once
int axera_run_sync(ggml_axera_kernel * k, std::vector<AX_ENGINE_IO_BUFFER_T> & in,
                   std::vector<AX_ENGINE_IO_BUFFER_T> & out);
// graph templates: the graph's name -> tensor map (cached in the context), the find by name, the output head
const std::map<std::string, ggml_tensor *> & axera_graph_names(ggml_backend_axera_context * ctx,
                                                               struct ggml_cgraph * cgraph);
ggml_tensor * axera_graph_find(const std::map<std::string, ggml_tensor *> & t, const char * name);
ggml_tensor * axera_graph_norm(const std::map<std::string, ggml_tensor *> & t);
ggml_tensor * axera_graph_head(const std::map<std::string, ggml_tensor *> & t);
const float * axera_final_norm(const ggml_tensor * mul, int H, float * eps);  // the final norm's weight and eps
void axera_output_norm(const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & h, int T);
void axera_output_head(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph,
                       const std::map<std::string, ggml_tensor *> & t, const std::vector<float> & h, int T,
                       const ggml_backend_axera_buffer_context * lm_q, const ggml_backend_axera_buffer_context * lm_s,
                       const ggml_backend_axera_buffer_context * lm_bf);
enum ggml_status axera_q35_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph);
bool axera_vit_graph(const struct ggml_cgraph * cgraph);
enum ggml_status axera_vit_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph);
bool axera_audio_graph(const struct ggml_cgraph * cgraph);
enum ggml_status axera_audio_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph);
// a tensor the model loader set whole: the audio encoder converts its layer matrices to bf16 now
void axera_audio_on_set(const ggml_tensor * t);
bool axera_llama_graph(const struct ggml_cgraph * cgraph);
enum ggml_status axera_gen_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph);
void axera_cpu_compute(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph, int i0, int i1, const char * what);
void axera_cpu_compute_nodes(ggml_backend_axera_context * ctx, struct ggml_cgraph * cgraph, ggml_tensor ** nodes, int n,
                             const char * what);

// GGML_AXERA_PROF=1: the host time between two template graphs split at the backend's entry points, for gaps under
// 50 ms (decode steps): graph end -> first get_tensor (llama.cpp after compute), the get_tensor calls, their end ->
// first set_tensor (sampling, the next graph's build and allocation), first set_tensor -> compute (inputs)
struct axera_host_phase {
    using clk = std::chrono::steady_clock;
    clk::time_point end, get0, get1, set0;
    void add(const char * what, clk::time_point a, clk::time_point b) {
        auto & e = g_axera_prof.t[what];
        e.first += std::chrono::duration<double, std::milli>(b - a).count();
        e.second++;
    }
    void on_get(clk::time_point a, clk::time_point b) {
        if (g_axera_prof.on && end.time_since_epoch().count()) {
            get0 = get0 > end ? get0 : a;
            get1 = b;
        }
    }
    void on_set() {
        if (g_axera_prof.on && end.time_since_epoch().count() && !(set0 > end)) {
            set0 = clk::now();
        }
    }
    void on_compute() {  // at the start of a graph
        const clk::time_point now = clk::now();
        if (!g_axera_prof.on || !end.time_since_epoch().count() || now - end > std::chrono::milliseconds(50)) {
            return;
        }
        std::lock_guard<std::mutex> lock(g_axera_prof.mu);
        const bool got = get0 > end, set = set0 > end && (!got || set0 > get1);
        const clk::time_point a = got ? get1 : end, s = set ? set0 : now;
        add("host 0: decode gap (all)", end, now);
        if (got) {
            add("host 1: graph end -> get_tensor", end, get0);
            add("host 2: get_tensor calls", get0, get1);
        }
        add("host 3: -> first set_tensor", a, s);
        add("host 4: first set_tensor -> graph", s, now);
    }
};
inline axera_host_phase g_axera_host;

//
// kernels
//

struct ggml_axera_kernel {
    AX_ENGINE_HANDLE      handle = nullptr;
    AX_ENGINE_IO_INFO_T * info   = nullptr;
    int                   cores   = 1;  // the build's core count (<name>_npu<cores>)
    // a single-core build loaded for lack of the multi-core one: warned at its first run (axera_run_sync),
    // not at load (a template also loads unpacked kernels for their IO only, running the packed ones)
    std::string           one_core;
    // the model file's bytes, kept for the handle's lifetime: the AX8860 engine of SDK V0.4.0 keeps pointers into
    // them (a later CreateHandle reads the names of the models before it: freed bytes crash it)
    std::vector<char>     model;

    ~ggml_axera_kernel() {
        if (handle) {
            AX_ENGINE_DestroyHandle(handle);
        }
    }

    int input(const char * name) const {
        for (uint32_t i = 0; i < info->nInputSize; ++i) {
            if (strcmp(info->pInputs[i].pName, name) == 0) {
                return (int) i;
            }
        }
        return -1;
    }

    int output(const char * name) const {
        for (uint32_t i = 0; i < info->nOutputSize; ++i) {
            if (strcmp(info->pOutputs[i].pName, name) == 0) {
                return (int) i;
            }
        }
        return -1;
    }
};

struct ggml_axera_kernel_registry {
    std::mutex                                                mutex;
    std::string                                               dir;
    std::map<int64_t, std::set<int64_t>>                      mul_mat;      // bf16: K -> available weight-row counts C
    std::map<int64_t, std::set<int64_t>>                      mul_mat_mx8;  // MXFP8 weights (x, w, s -> out)
    std::map<std::string, std::set<int64_t>>                  ctx;  // <name>_c<LMAX>.axmodel: name -> contexts
    std::set<std::string>                                     reported;
    std::map<std::string, std::unique_ptr<ggml_axera_kernel>> loaded;

    std::map<std::string, std::set<int>>                      files;  // base name -> core counts (<name>_npu<N>.axmodel)

    ggml_axera_kernel_registry() {
        const char * e = std::getenv("GGML_AXERA_KERNEL_DIR");
        dir = e ? e : "./axera_kernels";
        if (DIR * d = opendir(dir.c_str())) {
            while (dirent * f = readdir(d)) {
                const std::string n  = f->d_name;
                const size_t      us = n.rfind("_npu");
                int               cores, len = 0;
                if (us != std::string::npos && sscanf(n.c_str() + us, "_npu%d.axmodel%n", &cores, &len) == 1 &&
                    us + len == n.size()) {
                    files[n.substr(0, us)].insert(cores);
                }
            }
            closedir(d);
        }
        for (const auto & f : files) {  // the kernels this core count runs: shapes by family
            const std::string & n = f.first;
            long long           c, k;
            int                 len = 0;
            if (file(n).empty()) {
                continue;
            }
            if (sscanf(n.c_str(), "mul_mat_bf16_n%lld_k%lld%n", &c, &k, &len) == 2 && (size_t) len == n.size()) {
                mul_mat[k].insert(c);
            }
            len = 0;
            if (sscanf(n.c_str(), "mul_mat_mx8_n%lld_k%lld%n", &c, &k, &len) == 2 && (size_t) len == n.size()) {
                mul_mat_mx8[k].insert(c);
            }
            const size_t us = n.rfind("_c");
            len             = 0;
            if (us != std::string::npos && sscanf(n.c_str() + us, "_c%lld%n", &c, &len) == 1 && us + len == n.size()) {
                ctx[n.substr(0, us)].insert(c);
            }
        }
    }

    // the file of a kernel: <name>_npu<GGML_AXERA_CORES> (all cores), else its single-core build <name>_npu1; empty if
    // neither exists
    std::string file(const std::string & name) const {
        auto it = files.find(name);
        if (it == files.end()) {
            return "";
        }
        const int cores = it->second.count(axera_cores()) ? axera_cores() : it->second.count(1) ? 1 : 0;
        return cores ? name + "_npu" + std::to_string(cores) : "";
    }

    bool has(const std::string & name) const { return !file(name).empty(); }

    std::string path(const std::string & name) const {
        const std::string f = file(name);
        return dir + "/" + (f.empty() ? name + "_npu1" : f) + ".axmodel";
    }

    // the smallest context >= n among the <name>_c<LMAX> kernels, 0 if none
    int64_t ctx_for(const std::string & name, int64_t n) const {
        auto it = ctx.find(name);
        if (it == ctx.end()) {
            return 0;
        }
        auto c = it->second.lower_bound(n);
        return c == it->second.end() ? 0 : *c;
    }

    // the largest context <= n among the <name>_c<LMAX> kernels, 0 if none
    int64_t ctx_at_most(const std::string & name, int64_t n) const {
        auto it = ctx.find(name);
        if (it == ctx.end()) {
            return 0;
        }
        auto c = it->second.upper_bound(n);
        return c == it->second.begin() ? 0 : *std::prev(c);
    }

    // weight rows per call for a [N, K] weight: N itself, else the largest available C dividing N; 0 if none
    int64_t mul_mat_chunk(int64_t n, int64_t k, bool mx8 = false) {
        std::lock_guard<std::mutex> lock(mutex);
        auto & kernels = mx8 ? mul_mat_mx8 : mul_mat;
        auto   it      = kernels.find(k);
        if (it != kernels.end()) {
            for (auto c = it->second.rbegin(); c != it->second.rend(); ++c) {
                if (*c <= n && n % *c == 0) {
                    return *c;
                }
            }
        }
        char key[64];
        snprintf(key, sizeof(key), "%s_n%" PRId64 "_k%" PRId64, mx8 ? "mx8" : "bf16", n, k);
        if (axera_debug() && reported.insert(key).second) {
            GGML_LOG_INFO("%s: no kernel mul_mat_%s in %s\n", __func__, key, dir.c_str());
        }
        return 0;
    }

    ggml_axera_kernel * get(const std::string & name) {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = loaded.find(name);
        if (it != loaded.end()) {
            return it->second.get();
        }
        std::ifstream f(path(name), std::ios::binary);
        std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        auto k = std::make_unique<ggml_axera_kernel>();
        k->model = std::move(data);
        int ret = k->model.empty() ? -1 : AX_ENGINE_CreateHandle(&k->handle, k->model.data(), k->model.size());
        if (ret == 0) {
            ret = AX_ENGINE_CreateContext(k->handle);
        }
        if (ret == 0) {
            ret = AX_ENGINE_GetIOInfo(k->handle, &k->info);
        }
        if (ret != 0) {
            GGML_ABORT("%s: failed to load kernel %s (0x%x)\n", __func__, path(name).c_str(), ret);
        }
        if (axera_debug()) {  // (which kernels a model uses: GGML_AXERA_PRELOAD=0 GGML_AXERA_DEBUG=1)
            GGML_LOG_WARN("axera: kernel %s\n", path(name).c_str());
        }
        k->cores = atoi(file(name).c_str() + file(name).rfind("_npu") + 4);
        if (axera_cores() > 1 && file(name) == name + "_npu1") {  // no build for all cores: runs on one
            k->one_core = name;
        }
        return (loaded[name] = std::move(k)).get();
    }
};

struct ggml_backend_axera_context {
    std::shared_ptr<axera::model> model;  // the kernel directory's model adapter (a manifest), made by the first graph
    axera_cmm staging;    // bf16 activation row
    axera_cmm batch_out;  // f32 output rows of a batched GEMV call
    // the graph templates' state (created by their first graph)
    std::shared_ptr<axera_q35> q35;
    std::shared_ptr<axera_vit> vit;
    std::shared_ptr<axera_audio> audio;
    std::shared_ptr<axera_gen>   gen;
    // the template of the last graph (0 vision encoder, 1 audio encoder, 2 llama.cpp model, 3 other) and that graph
    struct {
        const struct ggml_cgraph * g     = nullptr;
        int                        n     = 0;
        const ggml_tensor *        first = nullptr, *last = nullptr;
        int                        kind  = 0;
    } graph_kind;
    // the CPU backend for the graph nodes a template leaves to it (all threads, one thread) and its threadpools
    ggml_backend_t               cpu[2]    = {};
    ggml_threadpool_t            cpu_tp[2] = {};
    void (*cpu_tp_free)(ggml_threadpool_t) = nullptr;
    // the graph templates' name -> tensor map of the last graph, kept while the graph keeps its nodes (llama.cpp
    // reuses the decode graph: rebuilding the map of a 35B-A3B graph takes ~10 ms on the A55)
    std::vector<uintptr_t>               graph_key;
    std::map<std::string, ggml_tensor *> graph_names;

    ~ggml_backend_axera_context() {
        for (int c = 0; c < 2; ++c) {
            if (cpu[c]) {
                ggml_backend_free(cpu[c]);
            }
            if (cpu_tp[c]) {
                cpu_tp_free(cpu_tp[c]);
            }
        }
    }

    ggml_backend_axera_buffer_context * staging_for(size_t size) {
        if (!staging || staging->size < size) {
            staging = axera_cmm_alloc(size);
        }
        return staging.get();
    }
};

