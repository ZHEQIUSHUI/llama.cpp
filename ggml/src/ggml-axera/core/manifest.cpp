// Axera NPU backend core: the kernel directory's manifest.json and its kernels

#include "manifest.h"

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

int kernel::output(const std::string & name) const {
    for (uint32_t i = 0; i < info->nOutputSize; ++i) {
        if (name == info->pOutputs[i].pName) {
            return (int) i;
        }
    }
    GGML_ABORT("axera: kernel %s (%s) has no output %s\n", desc->role.c_str(), desc->file.c_str(), name.c_str());
}

void kernel::run(std::vector<AX_ENGINE_IO_BUFFER_T> & in, std::vector<AX_ENGINE_IO_BUFFER_T> & out,
                 const char * what) const {
    GGML_ASSERT(in.size() == info->nInputSize && out.size() == info->nOutputSize);
    AX_ENGINE_IO_T io;
    memset(&io, 0, sizeof(io));
    io.pInputs     = in.data();
    io.nInputSize  = (uint32_t) in.size();
    io.pOutputs    = out.data();
    io.nOutputSize = (uint32_t) out.size();
    const int ret  = AX_ENGINE_RunSync(handle, &io);
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
            m->params[kind][name] = std::move(d);
        }
    }
    for (const auto & [role, k] : j.at("kernels").items()) {
        kernel_desc d;
        d.role = role;
        d.file = k.at("file").get<std::string>();
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
    for (const auto & [role, d] : m->kernels) {
        m->loaded[role] = load_kernel(dir, d);
    }
    GGML_LOG_INFO("axera: %s: model %s, context %lld, %zu kernels loaded in %.1f s\n", dir.c_str(), m->model.c_str(),
                  (long long) m->context, m->kernels.size(), (ggml_time_us() - t0) / 1e6);
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
