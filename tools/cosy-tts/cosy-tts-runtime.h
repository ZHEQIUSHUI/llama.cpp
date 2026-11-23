#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

using cosy_audio_chunk_callback = std::function<bool(const std::vector<uint8_t> & chunk, bool is_final)>;

// Standalone command-line entry point.  llama-tts calls this directly; there is
// no helper executable and no process hand-off.
int cosyvoice3_cli_main(int argc, char ** argv);

// Start the one in-process synthesis service.  The call returns only after all
// models, kernel sessions and the warm-up request are ready.
bool cosyvoice3_service_start(const std::vector<std::string> & args, std::string & error);

// Serialized request entry point used by llama-server.  `ref_wav` is a complete
// RIFF/WAVE byte string.  When `on_chunk` is set, audio is delivered as 16-bit
// little-endian PCM chunks; the final chunk has is_final=true.
bool cosyvoice3_service_request(const std::string & text,
                                const std::string & ref_text,
                                const std::vector<uint8_t> & ref_wav,
                                uint32_t seed, bool streaming,
                                const cosy_audio_chunk_callback & on_chunk,
                                std::string & error,
                                std::string * pcm_out = nullptr);

void cosyvoice3_service_stop();
bool cosyvoice3_service_available();
