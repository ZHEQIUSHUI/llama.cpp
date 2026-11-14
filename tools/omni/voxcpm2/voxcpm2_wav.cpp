#include "voxcpm2_wav.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>

namespace voxcpm2 {
namespace {

static uint16_t rd_u16(const uint8_t * p) {
    return static_cast<uint16_t>(p[0] | (p[1] << 8));
}

static uint32_t rd_u32(const uint8_t * p) {
    return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) | (static_cast<uint32_t>(p[3]) << 24);
}

// Decode one integer PCM sample of `bits` bits at `p` into [-1, 1].
static float decode_sample(const uint8_t * p, int bits) {
    switch (bits) {
        case 8:
            return (static_cast<int>(p[0]) - 128) / 128.0f;
        case 16: {
            int16_t v;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v) / 32768.0f;
        }
        case 24: {
            int32_t v = static_cast<int32_t>(p[0]) | (static_cast<int32_t>(p[1]) << 8) |
                        (static_cast<int32_t>(p[2]) << 16);
            if (v & 0x00800000) {
                v |= ~0x00FFFFFF;
            }
            return static_cast<float>(v) / 8388608.0f;
        }
        case 32: {
            int32_t v;
            std::memcpy(&v, p, sizeof(v));
            return static_cast<float>(v) / 2147483648.0f;
        }
        default:
            return 0.0f;
    }
}

static bool parse_wav(const uint8_t * data, size_t size, std::vector<float> & pcm, int & sample_rate) {
    if (size < 12 || std::memcmp(data, "RIFF", 4) != 0 || std::memcmp(data + 8, "WAVE", 4) != 0) {
        return false;
    }
    int16_t  audio_format    = 0;
    int16_t  num_channels    = 0;
    int16_t  bits_per_sample = 0;
    int32_t  rate            = 0;
    uint32_t data_size       = 0;
    const uint8_t * data_ptr = nullptr;

    size_t off = 12;
    while (off + 8 <= size) {
        const uint8_t * id   = data + off;
        uint32_t        csz  = rd_u32(data + off + 4);
        size_t          body = off + 8;
        if (body > size || csz > size - body) {
            return false;
        }
        if (std::memcmp(id, "fmt ", 4) == 0) {
            if (csz < 16) {
                return false;
            }
            audio_format    = static_cast<int16_t>(rd_u16(data + body));
            num_channels    = static_cast<int16_t>(rd_u16(data + body + 2));
            rate            = static_cast<int32_t>(rd_u32(data + body + 4));
            bits_per_sample = static_cast<int16_t>(rd_u16(data + body + 14));
        } else if (std::memcmp(id, "data", 4) == 0) {
            data_size = csz;
            data_ptr  = data + body;
            // WAV data may be the last chunk; stop scanning.
            break;
        }
        off = body + csz + (csz & 1u);
    }

    if (audio_format != 1 || num_channels < 1 || rate <= 0 || bits_per_sample <= 0 || bits_per_sample % 8 != 0 ||
        data_ptr == nullptr || data_size == 0) {
        return false;
    }
    const int bytes_per_sample = bits_per_sample / 8;
    if (bytes_per_sample != 1 && bytes_per_sample != 2 && bytes_per_sample != 3 && bytes_per_sample != 4) {
        return false;
    }
    const size_t frame_bytes = static_cast<size_t>(bytes_per_sample) * static_cast<size_t>(num_channels);
    if (frame_bytes == 0) {
        return false;
    }
    const size_t frames = data_size / frame_bytes;
    if (frames == 0) {
        return false;
    }

    pcm.resize(frames);
    for (size_t i = 0; i < frames; ++i) {
        float sum = 0.0f;
        const uint8_t * frame = data_ptr + i * frame_bytes;
        for (int ch = 0; ch < num_channels; ++ch) {
            sum += decode_sample(frame + static_cast<size_t>(ch) * bytes_per_sample, bits_per_sample);
        }
        pcm[i] = sum / static_cast<float>(num_channels);
    }
    sample_rate = rate;
    return true;
}

}  // namespace

bool load_wav_file(const std::string & path, std::vector<float> & pcm, int & sample_rate) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        return false;
    }
    const std::streamsize bytes = f.tellg();
    if (bytes <= 0) {
        return false;
    }
    std::vector<uint8_t> buf(static_cast<size_t>(bytes));
    f.seekg(0);
    f.read(reinterpret_cast<char *>(buf.data()), bytes);
    if (!f) {
        return false;
    }
    return load_wav_memory(buf.data(), buf.size(), pcm, sample_rate);
}

bool load_wav_memory(const uint8_t * data, size_t size, std::vector<float> & pcm, int & sample_rate) {
    return parse_wav(data, size, pcm, sample_rate);
}

bool write_wav_file(const std::string & path, const std::vector<float> & pcm, int sample_rate) {
    std::ofstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    const int32_t n_samples = static_cast<int32_t>(pcm.size());
    const int32_t data_size = n_samples * 2;
    const int32_t chunk     = 36 + data_size;
    auto w32 = [&](int32_t v) { f.write(reinterpret_cast<const char *>(&v), 4); };
    auto w16 = [&](int16_t v) { f.write(reinterpret_cast<const char *>(&v), 2); };
    f.write("RIFF", 4);
    w32(chunk);
    f.write("WAVE", 4);
    f.write("fmt ", 4);
    w32(16);
    w16(1);
    w16(1);
    w32(sample_rate);
    w32(sample_rate * 2);
    w16(2);
    w16(16);
    f.write("data", 4);
    w32(data_size);
    for (float v : pcm) {
        v = std::max(-1.0f, std::min(1.0f, v));
        w16(static_cast<int16_t>(v * 32767.0f));
    }
    return f.good();
}

}  // namespace voxcpm2
