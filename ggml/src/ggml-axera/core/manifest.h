// Axera NPU backend core: the kernel directory's manifest.json and its kernels
//
// The manifest is the only contract between the kernels (ax-dsl-kernels, tools/manifest) and this backend:
//   model     the model adapter that runs the directory (models/<model>)
//   context   the KV cells the attention kernels hold
//   params    per kind (gdn, attn, experts, <kind>_bf16), each layer parameter the kernels take: its GGUF tensor
//             (within blk.<i>.), the conversion from it and its kernel form; kernel inputs and pack members name
//             theirs (param, kind)
//   kernels   by role, the kernels of the model's path: file, IO table, packs and their members (packed kernels)
// Every kernel listed is loaded at the runtime's start; nothing else in the directory is read. A missing file, an IO
// table that differs from the manifest's or a role the adapter asks for and the manifest lacks stops the process.
#pragma once

#include "runtime.h"

#include <map>
#include <string>
#include <vector>

namespace axera {

struct io_desc {
    std::string          name;
    std::string          param, kind;  // a layer parameter input: the parameter and its kind (params), else empty
    uint32_t             bytes = 0;
    int                  dtype = 0;  // AX_ENGINE_DATA_TYPE_T
    std::vector<int64_t> shape;
};

// a parameter in a pack: [offset, offset + bytes) of pack `pack`, layer `layer` of the kernel's layers (-1: one layer)
struct pack_member {
    std::string name, pack, param, kind;  // kind: of params
    int         layer = -1;
    size_t      offset = 0, bytes = 0;
    // column strips: the [rows][cols] parameter stored as strips of width w, one after another
    int64_t strip_rows = 0, strip_cols = 0, strip_w = 0;
    // column blocks: byte ranges [a, b) of each row (row_bytes each, rows of them), each range for all rows in turn
    size_t                                   blk_rows = 0, blk_row_bytes = 0;
    std::vector<std::pair<size_t, size_t>>   ranges;
};

struct param_desc {
    std::string          gguf;     // tensor name within blk.<i>.
    std::string          convert;  // tensor, mxfp8, mxfp8_scales, mxfp4, mxfp4_scales, log_neg, bf16_taps_reversed,
                                   // mxfp4_planar_codes, mxfp4_planar_scales
    uint32_t             bytes = 0;
    int                  dtype = 0;
    std::vector<int64_t> shape;
};

struct kernel_desc {
    std::string                   role, file;
    std::vector<io_desc>          in, out;
    std::map<std::string, size_t> packs;  // pack input -> bytes
    std::vector<pack_member>      members;
};

// a loaded kernel
struct kernel {
    const kernel_desc *   desc   = nullptr;
    AX_ENGINE_HANDLE      handle = nullptr;
    AX_ENGINE_IO_INFO_T * info   = nullptr;
    std::vector<char>     model;  // the axmodel's bytes, kept for the handle's lifetime

    ~kernel();
    int  input(const std::string & name) const;   // index, aborts if absent
    int  output(const std::string & name) const;  // index, aborts if absent
    int  find_input(const std::string & name) const;  // index, -1 if absent
    uint32_t in_bytes(const std::string & name) const { return info->pInputs[input(name)].nSize; }
    uint32_t out_bytes(const std::string & name) const { return info->pOutputs[output(name)].nSize; }
    // one call; aborts with `what` and the IO on failure
    void run(std::vector<AX_ENGINE_IO_BUFFER_T> & in, std::vector<AX_ENGINE_IO_BUFFER_T> & out, const char * what) const;
};

struct manifest {
    std::string dir, model;
    int64_t     context = 0;
    int         cores   = 0;
    std::map<std::string, std::map<std::string, param_desc>> params;  // layer kind -> parameter -> description
    std::map<std::string, kernel_desc>                       kernels;  // role -> description

    // the role's kernel (loaded at the runtime's start); aborts if the manifest has no such role
    kernel & get(const std::string & role) const;
    bool     has(const std::string & role) const { return kernels.count(role) > 0; }
    const param_desc & param(const std::string & kind, const std::string & name) const;

    std::map<std::string, std::unique_ptr<kernel>> loaded;
};

// the manifest of GGML_AXERA_KERNEL_DIR (default ./axera_kernels), its kernels loaded; nullptr if the directory has no
// manifest.json
const manifest * kernels();

}  // namespace axera
