// Axera NPU backend core: layer parameters from the GGUF tensors, in the forms the manifest gives, and the packs of
// packed kernels filled from them
#pragma once

#include "manifest.h"
#include "quant.h"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace axera {

class params {
  public:
    explicit params(const manifest & m) : m(m) {}

    // the kernel IO of parameter `name` of kind `kind` (manifest params) of layer il (GGUF blk.<il>), converted once
    // from the graph's tensors `t` and kept until release()
    AX_ENGINE_IO_BUFFER_T get(const std::map<std::string, ggml_tensor *> & t, int il, const std::string & kind,
                              const std::string & name);

    // pack `pack` of kernel kd for the layers il0 .. (members of layer j: il0 + j) in a new buffer; packs of the same
    // members at the same offsets (kernels of equal layouts) are made once and shared
    const axera_cmm & pack(const std::map<std::string, ggml_tensor *> & t, const kernel_desc & kd, const std::string & pack,
                           int il0);

    // the converted parameters no IO in `live` points into, freed (the packs and the experts stay): once every
    // kernel is bound, those read only through packs
    void release(const std::vector<const std::vector<AX_ENGINE_IO_BUFFER_T> *> & live);

  private:
    const manifest & m;
    struct mx_pair {
        axera_cmm q, s;
    };
    std::map<std::pair<const ggml_tensor *, int>, mx_pair> mx;    // (tensor, 8 / 4) -> MXFP8 / MXFP4
    std::map<const ggml_tensor *, axera_cmm>               bf16;  // bf16 copies
    std::map<std::pair<int, std::string>, axera_cmm>       own;   // (layer, kind / name) -> converted copy
    std::map<std::string, axera_cmm>                       packs; // layout signature -> pack
    struct experts {  // a layer's routed experts as planes (in CMM, repacked in place): gate-up, down
        ggml_backend_axera_buffer_context * b[2] = {};
        char *                              base[2] = {};
        int64_t                             n = 0;
        size_t                              data[2] = {}, scale[2] = {};
    };
    std::map<int, experts> exps;
    std::vector<axera_cmm> exp_own;  // gate-up experts whose gate and up are not back to back

    const experts & experts_of(const std::map<std::string, ggml_tensor *> & t, int il);
    AX_ENGINE_IO_BUFFER_T convert(const std::map<std::string, ggml_tensor *> & t, int il, const std::string & kind,
                                  const std::string & name, const param_desc & p);
};

}  // namespace axera
