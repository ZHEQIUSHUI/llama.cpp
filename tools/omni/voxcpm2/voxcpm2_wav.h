#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace voxcpm2 {

// Decode a mono/stereo PCM WAV (8/16/24/32-bit integer) into mono float32.
// Returns false on unsupported/invalid input. On success `sample_rate` is set.
bool load_wav_file(const std::string & path, std::vector<float> & pcm, int & sample_rate);
bool load_wav_memory(const uint8_t * data, size_t size, std::vector<float> & pcm, int & sample_rate);

// Write mono float32 as 16-bit PCM WAV.
bool write_wav_file(const std::string & path, const std::vector<float> & pcm, int sample_rate);

}  // namespace voxcpm2
