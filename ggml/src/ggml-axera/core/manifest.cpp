// Axera NPU backend core: the kernel directory's manifest.json and its kernels

#include "manifest.h"

#include <map>
#include <set>

#include <nlohmann/json.hpp>

#include <fstream>
#include <iterator>

namespace axera {

kernel::~kernel() {
    if (handle) {
        AX_ENGINE_DestroyHandle(handle);
    }
}

int kernel::find_input(const std::string & name) const {
    for (uint32_t i = 0; i < info->nInputSize; ++i) {
        if (name == info->pInputs[i].pName) {
            return (int) i;
        }
    }
    return -1;
}

int kernel::input(const std::string & name) const {
    const int i = find_input(name);
    if (i < 0) {
        GGML_ABORT("axera: kernel %s (%s) has no input %s\n", desc->role.c_str(), desc->file.c_str(), name.c_str());
    }
    return i;
}

int kernel::find_output(const std::string & name) const {
    for (uint32_t i = 0; i < info->nOutputSize; ++i) {
        if (name == info->pOutputs[i].pName) {
            return (int) i;
        }
    }
    return -1;
}

int kernel::output(const std::string & name) const {
    const int i = find_output(name);
    if (i < 0) {
        GGML_ABORT("axera: kernel %s (%s) has no output %s\n", desc->role.c_str(), desc->file.c_str(), name.c_str());
    }
    return i;
}

void kernel::run(std::vector<AX_ENGINE_IO_BUFFER_T> & in, std::vector<AX_ENGINE_IO_BUFFER_T> & out,
                 const char * what) const {
    GGML_ASSERT(in.size() == info->nInputSize && out.size() == info->nOutputSize);
    axera_timer tpre("kernel: host pre");
    AX_ENGINE_IO_T io;
    memset(&io, 0, sizeof(io));
    io.pInputs     = in.data();
    io.nInputSize  = (uint32_t) in.size();
    io.pOutputs    = out.data();
    io.nOutputSize = (uint32_t) out.size();
    // debug: GGML_AXERA_DUMP_ROLE=<role> GGML_AXERA_DUMP_DIR=<dir>: the role's first call's inputs as <dir>/<input>.bin,
    // then exit before running it (to replay it alone)
    static const char * dump_role = getenv("GGML_AXERA_DUMP_ROLE");
    static const char * dump_at_s = getenv("GGML_AXERA_DUMP_CALL");
    static const uint64_t dump_at = dump_at_s ? strtoull(dump_at_s, nullptr, 10) : 0;
    static std::map<std::string, uint64_t> dump_calls;
    const uint64_t call_no = dump_calls[desc->role]++;
    static const std::string dump_roles = [] {
        const char * e = getenv("GGML_AXERA_DUMP_ROLES");
        return e ? std::string(e) : std::string();
    }();
    static const std::string dump_only = [] {
        const char * e = getenv("GGML_AXERA_DUMP_ONLY");
        return e ? std::string(",") + e + "," : std::string();
    }();
    bool dump_role_at_match = false;
    if (!dump_roles.empty()) {
        const std::string a   = "," + dump_roles + ",";
        const size_t      pos = a.find("," + desc->role + ":");
        if (pos != std::string::npos) {
            const size_t s0 = pos + desc->role.size() + 2;
            dump_role_at_match = call_no == strtoull(a.c_str() + s0, nullptr, 10);
        }
    }
    if ((dump_role && desc->role == dump_role && call_no == dump_at) || dump_role_at_match) {
        const char * dir = getenv("GGML_AXERA_DUMP_DIR");
        for (size_t i = 0; i < in.size(); ++i) {
            const std::string name = info->pInputs[i].pName;
            if (!dump_only.empty() && dump_only.find("," + name + ",") == std::string::npos) {
                continue;
            }
            const std::string p = std::string(dir ? dir : ".") + "/" + name + ".bin";
            FILE *            f = fopen(p.c_str(), "wb");
            GGML_ASSERT(f && in[i].pVirAddr);
            fwrite(in[i].pVirAddr, 1, info->pInputs[i].nSize, f);
            fclose(f);
        }
        fprintf(stderr, "axera: %s: %zu inputs of %s dumped to %s\n", what, in.size(), desc->role.c_str(), dir ? dir : ".");
        if (!getenv("GGML_AXERA_DUMP_NOEXIT")) {
            exit(0);
        }
    }
    static const bool io_stats = getenv("GGML_AXERA_IO_STATS") != nullptr;
    if (io_stats) {
        static std::set<std::string> seen;
        if (seen.insert(desc->role).second) {
            auto align_of = [](uint64_t a) { return a ? (a & (~a + 1)) : 0ull; };
            unsigned min_a = ~0u, n4k = 0, nbad = 0;
            for (const auto & b : in) {
                const unsigned a = (unsigned) std::min<uint64_t>(align_of(b.phyAddr), 1u << 20);
                min_a = std::min(min_a, a);
                n4k += (b.phyAddr & 4095) == 0;
                nbad += (b.phyAddr & 127) != 0;
            }
            for (const auto & b : out) {
                const unsigned a = (unsigned) std::min<uint64_t>(align_of(b.phyAddr), 1u << 20);
                min_a = std::min(min_a, a);
                n4k += (b.phyAddr & 4095) == 0;
                nbad += (b.phyAddr & 127) != 0;
            }
            fprintf(stderr, "axera io stats: %-16s in %zu out %zu min-align %u, 4k %u/%zu, <128B %u\n", desc->role.c_str(),
                    in.size(), out.size(), min_a, n4k, in.size() + out.size(), nbad);
        }
    }
    // debug: GGML_AXERA_REPEAT_ROLE=<role> GGML_AXERA_REPEAT_CALL=<n> GGML_AXERA_REPEAT_N=<runs>: time the real call
    // and then repeat it with the same IO and buffers, then exit before the mutated state is used. This distinguishes a
    // kernel's real-input cost from one call's host/runtime overhead (the repeat keeps the prepared IO arrays).
    static const char * repeat_role = getenv("GGML_AXERA_REPEAT_ROLE");
    static const uint64_t repeat_at_s = [] {
        const char * e = getenv("GGML_AXERA_REPEAT_CALL");
        return e ? strtoull(e, nullptr, 10) : 0;
    }();
    static const int repeat_n = [] {
        const char * e = getenv("GGML_AXERA_REPEAT_N");
        return e ? atoi(e) : 100;
    }();
    if (repeat_role && desc->role == repeat_role && call_no == repeat_at_s) {
        AX_ENGINE_IO_T rio;
        memset(&rio, 0, sizeof(rio));
        rio.pInputs     = in.data();
        rio.nInputSize  = (uint32_t) in.size();
        rio.pOutputs    = out.data();
        rio.nOutputSize = (uint32_t) out.size();
        const auto t0 = std::chrono::steady_clock::now();
        int        r0 = AX_ENGINE_RunSync(handle, &rio);
        const auto t1 = std::chrono::steady_clock::now();
        for (int i = 0; i < repeat_n && r0 == 0; ++i) {
            r0 = AX_ENGINE_RunSync(handle, &rio);
        }
        const auto t2 = std::chrono::steady_clock::now();
        const double first = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const double rest  = std::chrono::duration<double, std::milli>(t2 - t1).count();
        fprintf(stderr, "axera repeat: %s call %llu: first %.3f ms, %.3f ms/run over %d runs%s (0x%x)\n",
                desc->role.c_str(), (unsigned long long) call_no, first, repeat_n ? rest / repeat_n : 0.0, repeat_n,
                r0 ? " FAILED" : "", r0);
        exit(r0 ? 1 : 0);
    }
    axera_thermal_guard();
    tpre.stop();
    axera_timer trun("kernel: RunSync");
    const int ret  = AX_ENGINE_RunSync(handle, &io);
    trun.stop();
    if (ret != 0) {  // the IO the kernel got (stderr: the log is asynchronous)
        for (int o = 0; o < 2; ++o) {
            const auto & bufs = o ? out : in;
            for (size_t i = 0; i < bufs.size(); ++i) {
                const AX_ENGINE_IOMETA_T & mt = o ? info->pOutputs[i] : info->pInputs[i];
                fprintf(stderr, "axera: RunSync 0x%x: %s %zu %s: %u bytes (bound %u at phy 0x%llx)\n", ret,
                        o ? "output" : "input", i, mt.pName, mt.nSize, bufs[i].nSize, (unsigned long long) bufs[i].phyAddr);
            }
        }
        GGML_ABORT("axera: %s: kernel %s failed (0x%x)\n", what, desc->role.c_str(), ret);
    }
}

kernel & manifest::get(const std::string & role) const {
    auto it = loaded.find(role);
    if (it == loaded.end()) {
        GGML_ABORT("axera: the manifest of %s (model %s) has no kernel for %s\n", dir.c_str(), model.c_str(), role.c_str());
    }
    return *it->second;
}

kernel & manifest::load_memory(const std::string & role, std::vector<char> model) const {
    GGML_ASSERT(!loaded.count(role));
    auto d   = std::make_unique<kernel_desc>();
    d->role  = role;
    d->file  = "<memory>";
    auto k   = std::make_unique<kernel>();
    k->desc  = d.get();
    k->model = std::move(model);
    int ret  = k->model.empty() ? -1 : AX_ENGINE_CreateHandle(&k->handle, k->model.data(), k->model.size());
    if (ret == 0) {
        ret = AX_ENGINE_CreateContext(k->handle);
    }
    if (ret == 0) {
        ret = AX_ENGINE_GetIOInfo(k->handle, &k->info);
    }
    if (ret != 0) {
        GGML_ABORT("axera: failed to load the in-memory kernel %s (0x%x)\n", role.c_str(), ret);
    }
    extra_desc.push_back(std::move(d));
    auto & out  = *k;
    loaded[role] = std::move(k);
    return out;
}

const param_desc & manifest::param(const std::string & kind, const std::string & name) const {
    auto k = params.find(kind);
    if (k == params.end() || !k->second.count(name)) {
        GGML_ABORT("axera: the manifest of %s has no %s parameter %s\n", dir.c_str(), kind.c_str(), name.c_str());
    }
    return k->second.at(name);
}

static std::vector<io_desc> read_io(const nlohmann::json & a) {
    std::vector<io_desc> v;
    for (const auto & e : a) {
        io_desc d;
        d.name  = e.at("name").get<std::string>();
        d.bytes = e.at("bytes").get<uint32_t>();
        d.dtype = e.at("dtype").get<int>();
        d.shape = e.at("shape").get<std::vector<int64_t>>();
        d.param = e.value("param", std::string());
        d.kind  = e.value("kind", std::string());
        v.push_back(std::move(d));
    }
    return v;
}

static std::unique_ptr<kernel> load_kernel(const std::string & dir, const kernel_desc & d) {
    const std::string path = dir + "/" + d.file;
    std::ifstream     f(path, std::ios::binary);
    auto              k = std::make_unique<kernel>();
    k->desc             = &d;
    k->model.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    int ret = k->model.empty() ? -1 : AX_ENGINE_CreateHandle(&k->handle, k->model.data(), k->model.size());
    if (ret == 0) {
        ret = AX_ENGINE_CreateContext(k->handle);
    }
    if (ret == 0) {
        ret = AX_ENGINE_GetIOInfo(k->handle, &k->info);
    }
    if (ret != 0) {
        GGML_ABORT("axera: failed to load kernel %s (%s, 0x%x)\n", d.role.c_str(), path.c_str(), ret);
    }
    // the kernel's IO table is the manifest's
    bool same = k->info->nInputSize == d.in.size() && k->info->nOutputSize == d.out.size();
    for (size_t i = 0; same && i < d.in.size(); ++i) {
        same = d.in[i].name == k->info->pInputs[i].pName && d.in[i].bytes == k->info->pInputs[i].nSize;
    }
    for (size_t i = 0; same && i < d.out.size(); ++i) {
        same = d.out[i].name == k->info->pOutputs[i].pName && d.out[i].bytes == k->info->pOutputs[i].nSize;
    }
    if (!same) {
        GGML_ABORT("axera: kernel %s (%s): its IO is not the manifest's\n", d.role.c_str(), path.c_str());
    }
    return k;
}

static std::unique_ptr<manifest> load_manifest(const std::string & dir) {
    std::ifstream f(dir + "/manifest.json");
    if (!f.good()) {
        return nullptr;
    }
    const nlohmann::json j = nlohmann::json::parse(f);
    auto                 m = std::make_unique<manifest>();
    m->dir                 = dir;
    if (j.at("format").get<int>() != 1) {
        GGML_ABORT("axera: %s/manifest.json: format %d, this backend reads format 1\n", dir.c_str(), j.at("format").get<int>());
    }
    m->model   = j.at("model").get<std::string>();
    m->context = j.at("context").get<int64_t>();
    m->cores   = j.at("cores").get<int>();
    for (const auto & [kind, ps] : j.at("params").items()) {
        for (const auto & [name, p] : ps.items()) {
            param_desc d;
            d.gguf    = p.at("gguf").get<std::string>();
            d.convert = p.at("convert").get<std::string>();
            d.bytes   = p.at("bytes").get<uint32_t>();
            d.dtype   = p.at("dtype").get<int>();
            d.shape   = p.at("shape").get<std::vector<int64_t>>();
            d.vperm   = p.value("vperm", std::string());
            m->params[kind][name] = std::move(d);
        }
    }
    for (const auto & [role, k] : j.at("kernels").items()) {
        kernel_desc d;
        d.role = role;
        d.file = k.at("file").get<std::string>();
        d.embedded = k.value("embedded", false);
        d.in   = read_io(k.at("in"));
        d.out  = read_io(k.at("out"));
        if (k.contains("packs")) {
            d.packs = k.at("packs").get<std::map<std::string, size_t>>();
            for (const auto & e : k.at("members")) {
                pack_member pm;
                pm.name   = e.at("name").get<std::string>();
                pm.pack   = e.at("pack").get<std::string>();
                pm.param  = e.at("param").get<std::string>();
                pm.kind   = e.at("kind").get<std::string>();
                pm.layer  = e.value("layer", -1);
                pm.offset = e.at("offset").get<size_t>();
                pm.bytes  = e.at("bytes").get<size_t>();
                if (e.contains("strips")) {
                    const auto s  = e.at("strips").get<std::vector<int64_t>>();
                    pm.strip_rows = s.at(0);
                    pm.strip_cols = s.at(1);
                    pm.strip_w    = s.at(2);
                    GGML_ASSERT(pm.strip_w > 0 && pm.strip_cols % pm.strip_w == 0 &&
                                pm.bytes % (size_t) (pm.strip_rows * pm.strip_cols) == 0);
                }
                if (e.contains("blocks")) {
                    const auto & b   = e.at("blocks");
                    pm.blk_rows      = b.at("rows").get<size_t>();
                    pm.blk_row_bytes = b.at("row_bytes").get<size_t>();
                    for (const auto & r : b.at("ranges")) {
                        pm.ranges.push_back({ r.at(0).get<size_t>(), r.at(1).get<size_t>() });
                    }
                    GGML_ASSERT(pm.blk_rows * pm.blk_row_bytes == pm.bytes);
                }
                GGML_ASSERT(d.packs.count(pm.pack) && pm.offset + pm.bytes <= d.packs.at(pm.pack));
                d.members.push_back(std::move(pm));
            }
        }
        m->kernels[role] = std::move(d);
    }
    const int64_t t0 = ggml_time_us();
    size_t n_embedded = 0;
    for (const auto & [role, d] : m->kernels) {
        if (d.embedded) {
            ++n_embedded;
            continue;
        }
        m->loaded[role] = load_kernel(dir, d);
    }
    GGML_LOG_INFO("axera: %s: model %s, context %lld, %zu kernels loaded, %zu embedded, in %.1f s\n", dir.c_str(),
                  m->model.c_str(), (long long) m->context, m->kernels.size() - n_embedded, n_embedded,
                  (ggml_time_us() - t0) / 1e6);
    return m;
}

const manifest * kernels() {
    static std::once_flag            once;
    static std::unique_ptr<manifest> m;
    std::call_once(once, [] {
        if (!axera_runtime_init()) {
            return;
        }
        const char * e = std::getenv("GGML_AXERA_KERNEL_DIR");
        m              = load_manifest(e ? e : "./axera_kernels");
    });
    return m.get();
}

}  // namespace axera
