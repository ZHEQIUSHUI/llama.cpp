// Axera NPU backend: the Qwen3-TTS codec decoder in one kernel call per window (role gen.codec), see codec.cpp
#pragma once

#include "../model.h"
#include "../../ops/ops.h"

#include <map>
#include <string>
#include <vector>

namespace axera {

class tts_codec {
  public:
    explicit tts_codec(const manifest & m);
    // a graph runs mtmd's code2wav (inp_codes -> out_audio, with the streaming state)
    static bool graph(const graph_names & names);
    // the graph's batch of frames: its samples into out_audio, state_out_tfm_pos written
    void run(const graph_names & names);

  private:
    const kernel &                     k;
    const void *                       bound = nullptr;  // the codebook's data: the model of the banks
    axera_cmm                          QW, UW, DW, CODES_B, POS, WAV;
    std::map<std::string, axera_cmm>   cur, nxt;  // the state at the window's start, the kernel's next state
    std::map<std::string, std::string> pair;      // a state input -> its next

    void                       pack(const graph_names & names);
    void                       bank(axera_cmm & c, const char * name, const std::vector<ggml_bf16_t> & v);
    void                       reset();
    static const ggml_tensor * T(const std::string & n, const graph_names & names);
};

}  // namespace axera
