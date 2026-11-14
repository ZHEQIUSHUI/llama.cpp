#pragma once

#include "llama.h"
#include "mtmd.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace mtmd_forced_aligner {

struct options {
    std::string transcript;
    std::string language = "English";
    int segment_ms = 80;
};

struct result {
    bool ok = false;
    int code = 500;
    std::string error;
    std::string json;
    int input_tokens = 0;
    int audio_tokens = 0;
    double forward_seconds = 0.0;
};

result run(llama_model * model, llama_context * ctx, mtmd_context * mctx,
           const void * audio, size_t audio_len, const options & opt);

} // namespace mtmd_forced_aligner
