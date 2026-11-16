// Fun-CosyVoice3-0.5B-2512 end-to-end runner.
//
// The Qwen2 backbone is evaluated through llama.cpp (CPU or the AXERA backend).
// The reference-audio encoders use the official ONNX graphs through ONNX
// Runtime on the host.  The flow-matching decoder and HiFT vocoder use the
// official ONNX graphs by default, or fixed-shape AXERA models with
// --flow-npu/--hift-npu.  The speech token table/head are stored in a small
// auxiliary GGUF next to the backbone GGUF.

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "gguf.h"
#include "mtmd-audio.h"
#include "cosy-tts-runtime.h"

#ifdef COSY_TTS_HAVE_AXERA
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <cmath>
#include <thread>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>

struct wav_data {
    std::vector<float> pcm;
    int sample_rate = 0;
};

static bool read_wav(const char * path, wav_data & out) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    char riff[4], wave[4];
    uint32_t riff_size = 0;
    if (fread(riff, 1, 4, f) != 4 || fread(&riff_size, 4, 1, f) != 1 || fread(wave, 1, 4, f) != 4 ||
        memcmp(riff, "RIFF", 4) || memcmp(wave, "WAVE", 4)) {
        fprintf(stderr, "%s: not a RIFF/WAVE file\n", path); fclose(f); return false;
    }
    uint16_t fmt = 0, channels = 0, bits = 0;
    uint32_t rate = 0;
    std::vector<uint8_t> data;
    while (true) {
        char id[4]; uint32_t size = 0;
        if (fread(id, 1, 4, f) != 4 || fread(&size, 4, 1, f) != 1) break;
        std::vector<uint8_t> chunk(size);
        if (fread(chunk.data(), 1, size, f) != size) break;
        if (!memcmp(id, "fmt ", 4)) {
            if (size < 16) break;
            memcpy(&fmt, chunk.data(), 2);
            memcpy(&channels, chunk.data() + 2, 2);
            memcpy(&rate, chunk.data() + 4, 4);
            memcpy(&bits, chunk.data() + 14, 2);
            if (fmt == 0xfffe && size >= 26) memcpy(&fmt, chunk.data() + 24, 2);
        } else if (!memcmp(id, "data", 4)) {
            data = std::move(chunk);
        }
        if (size & 1) fseek(f, 1, SEEK_CUR);
    }
    fclose(f);
    if (!channels || !rate || !bits || data.empty()) { fprintf(stderr, "%s: missing fmt/data\n", path); return false; }
    const int bytes = bits / 8;
    if (!bytes || (fmt != 1 && fmt != 3) || (fmt == 1 && bits != 8 && bits != 16 && bits != 24 && bits != 32) ||
        (fmt == 3 && bits != 32 && bits != 64)) {
        fprintf(stderr, "%s: unsupported wav fmt=%u bits=%u\n", path, fmt, bits); return false;
    }
    const size_t n = data.size() / (size_t) bytes;
    out.pcm.resize(n / channels);
    auto sample = [&](size_t j) -> float {
        const uint8_t * p8 = data.data() + j * (size_t) bytes;
        if (fmt == 3) {
            if (bits == 32) { float v; memcpy(&v, p8, 4); return v; }
            double v; memcpy(&v, p8, 8); return (float) v;
        }
        if (bits == 8)  return (p8[0] - 128) / 128.0f;
        if (bits == 16) { int16_t v; memcpy(&v, p8, 2); return v / 32768.0f; }
        if (bits == 24) {
            int32_t v = (int32_t) (((uint32_t) p8[2] << 16) | ((uint32_t) p8[1] << 8) | (uint32_t) p8[0]);
            if (v & 0x800000) v |= (int32_t) 0xff000000u;
            return (float) (v / 8388608.0);
        }
        int32_t v; memcpy(&v, p8, 4);
        return (float) ((double) v / 2147483648.0);
    };
    for (size_t i = 0; i < out.pcm.size(); i++) {
        float acc = 0;
        for (uint16_t c = 0; c < channels; c++) acc += sample(i * channels + c);
        out.pcm[i] = acc / channels;
    }
    out.sample_rate = (int) rate;
    return true;
}

static bool write_wav(const char * path, const std::vector<float> & pcm, int sample_rate) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    uint32_t data_size = (uint32_t) pcm.size() * 2;
    uint32_t riff_size = 36 + data_size;
    fwrite("RIFF", 1, 4, f); fwrite(&riff_size, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); uint32_t fmt_size = 16; fwrite(&fmt_size, 4, 1, f);
    uint16_t fmt = 1, ch = 1, align = 2, bits = 16; uint32_t rate = sample_rate, byte_rate = rate * 2;
    fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f); fwrite(&rate, 4, 1, f); fwrite(&byte_rate, 4, 1, f);
    fwrite(&align, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&data_size, 4, 1, f);
    for (float v : pcm) { int16_t s = (int16_t) std::lrint(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f); fwrite(&s, 2, 1, f); }
    fclose(f);
    return true;
}

static std::vector<float> resample_linear(const std::vector<float> & in, int sr_in, int sr_out) {
    if (sr_in == sr_out || in.empty()) return in;
    std::vector<float> out((size_t) std::llround((double) in.size() * sr_out / sr_in));
    for (size_t i = 0; i < out.size(); i++) {
        double x = (double) i * sr_in / sr_out;
        size_t i0 = (size_t) x, i1 = std::min(i0 + 1, in.size() - 1);
        double t = x - i0;
        out[i] = (float) ((1.0 - t) * in[i0] + t * in[i1]);
    }
    return out;
}

// Same default band-limited sinc interpolator as torchaudio.functional.resample
// (sinc_interp_hann, lowpass_filter_width=6, rolloff=0.99).
static std::vector<float> resample_sinc(const std::vector<float> & in, int sr_in, int sr_out) {
    if (sr_in == sr_out || in.empty()) return in;
    const int gcd = std::gcd(sr_in, sr_out);
    const int orig = sr_in / gcd, dst = sr_out / gcd;
    const float base = std::min(orig, dst) * 0.99f;
    const int width = (int) std::ceil(6.0 * orig / base);
    const int tap = 2 * width + orig;
    std::vector<float> kernel((size_t) dst * tap);
    for (int phase = 0; phase < dst; phase++) {
        for (int k = 0; k < tap; k++) {
            float t = -phase / (float) dst + (k - width) / (float) orig;
            t = std::max(-6.0f, std::min(6.0f, t * base));
            const float window = std::cos(t * (float) M_PI / 6.0f / 2.0f);
            t *= (float) M_PI;
            const float sinc = t == 0.0f ? 1.0f : std::sin(t) / t;
            kernel[(size_t) phase * tap + k] = sinc * window * window * base / orig;
        }
    }

    std::vector<float> pad((size_t) in.size() + 2 * width + orig, 0.0f);
    std::copy(in.begin(), in.end(), pad.begin() + width);
    const size_t out_len = (size_t) std::ceil((double) dst * in.size() / orig);
    std::vector<float> out(out_len);
    for (size_t j = 0; j < out_len; j++) {
        const float * k = kernel.data() + (size_t) (j % dst) * tap;
        const float * x = pad.data() + (j / dst) * orig;
        double sum = 0.0;
        for (int i = 0; i < tap; i++) sum += (double) k[i] * x[i];
        out[j] = (float) sum;
    }
    return out;
}

static std::vector<float> load_f32(const std::string & path, size_t expected = 0) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    size_t n = (size_t) f.tellg() / sizeof(float);
    if (expected && n != expected) { fprintf(stderr, "%s: expected %zu floats, got %zu\n", path.c_str(), expected, n); std::exit(1); }
    std::vector<float> v(n); f.seekg(0); f.read((char *) v.data(), n * sizeof(float)); return v;
}

static std::vector<float> load_f32_optional(const std::string & path, size_t expected) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return {};
    const size_t n = (size_t) f.tellg() / sizeof(float);
    if (n != expected) {
        fprintf(stderr, "%s: expected %zu floats, got %zu; ignoring\n", path.c_str(), expected, n);
        return {};
    }
    std::vector<float> v(n);
    f.seekg(0);
    if (!f.read((char *) v.data(), n * sizeof(float))) return {};
    return v;
}

// Read the int32 vector written by ref/run_reference.py's --codes file, or the
// raw little-endian int32 form emitted by --dump-codes.
static bool load_codes(const std::string & path, std::vector<int> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { fprintf(stderr, "cannot open codes %s\n", path.c_str()); return false; }
    std::vector<char> buf((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (buf.size() >= 10 && !memcmp(buf.data(), "\x93NUMPY", 6)) {
        const int major = (uint8_t) buf[6];
        size_t header_len = 0, header_off = 0;
        if (major == 1) {
            header_len = (uint8_t) buf[8] | ((size_t) (uint8_t) buf[9] << 8);
            header_off = 10;
        } else if (major == 2 || major == 3) {
            header_len = (size_t) (uint8_t) buf[8] | ((size_t) (uint8_t) buf[9] << 8) |
                         ((size_t) (uint8_t) buf[10] << 16) | ((size_t) (uint8_t) buf[11] << 24);
            header_off = 12;
        } else {
            fprintf(stderr, "%s: unsupported npy version %d\n", path.c_str(), major);
            return false;
        }
        if (header_off + header_len > buf.size()) { fprintf(stderr, "%s: truncated npy header\n", path.c_str()); return false; }
        std::string hdr(buf.data() + header_off, header_len);
        const bool i64 = hdr.find("'<i8'") != std::string::npos || hdr.find("\"<i8\"") != std::string::npos;
        const bool i32 = hdr.find("'<i4'") != std::string::npos || hdr.find("\"<i4\"") != std::string::npos;
        if (!i64 && !i32) { fprintf(stderr, "%s: codes npy must be int32 or int64\n", path.c_str()); return false; }
        const size_t elem = i64 ? 8 : 4;
        const size_t data_off = header_off + header_len;
        if ((buf.size() - data_off) % elem) { fprintf(stderr, "%s: malformed npy payload\n", path.c_str()); return false; }
        const size_t n = (buf.size() - data_off) / elem;
        out.resize(n);
        for (size_t i = 0; i < n; i++) {
            if (i64) {
                int64_t v = 0;
                memcpy(&v, buf.data() + data_off + i * elem, sizeof(v));
                out[i] = (int) v;
            } else {
                int32_t v = 0;
                memcpy(&v, buf.data() + data_off + i * elem, sizeof(v));
                out[i] = v;
            }
        }
        return true;
    }
    if (buf.size() % 4) { fprintf(stderr, "%s: raw codes must be int32\n", path.c_str()); return false; }
    const size_t n = buf.size() / 4;
    out.resize(n);
    for (size_t i = 0; i < n; i++) memcpy(&out[i], buf.data() + i * 4, 4);
    return true;
}

static bool read_binary_file(const std::string & path, std::vector<uint8_t> & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !out.empty();
}

static bool write_binary_file(const std::string & path, const std::vector<uint8_t> & data) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write((const char *) data.data(), (std::streamsize) data.size());
    return (bool) f;
}

static bool flow_silent_token(int code, int & run) {
    static const int silent[] = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323};
    bool found = false;
    for (int s : silent) found = found || code == s;
    if (found) run++; else run = 0;
    return found && run > 5;
}

static std::vector<int> filter_flow_silent_tokens(const std::vector<int> & codes) {
    std::vector<int> out;
    out.reserve(codes.size());
    int run = 0;
    for (int tok : codes) {
        if (!flow_silent_token(tok, run)) out.push_back(tok);
    }
    return out;
}

// Split a UTF-8 request into synthesis segments that fit the fixed flow
// window.  The estimator is not a sliding-window model: every window must
// start from the reference prompt at position zero, so a long utterance is
// split at punctuation and each piece is synthesized as its own sentence
// with the same reference.  Costs are conservative speech tokens per
// character, measured on the Chinese and English evaluation sets.
static std::vector<std::string> split_text_segments(const std::string & text, int gen_cap) {
    struct point { uint32_t cp; size_t off; size_t len; };
    std::vector<point> cps;
    for (size_t i = 0; i < text.size();) {
        const unsigned char * p = (const unsigned char *) text.data() + i;
        uint32_t cp = p[0];
        size_t len = 1;
        if (p[0] >= 0xF0) { cp = p[0] & 0x07; len = 4; }
        else if (p[0] >= 0xE0) { cp = p[0] & 0x0F; len = 3; }
        else if (p[0] >= 0xC0) { cp = p[0] & 0x1F; len = 2; }
        if (i + len > text.size()) len = 1;
        if (len > 1) {
            for (size_t k = 1; k < len; k++) cp = (cp << 6) | (p[k] & 0x3F);
        }
        cps.push_back({cp, i, len});
        i += len;
    }
    if (gen_cap <= 0 || cps.empty()) return {text};
    // gen_cap already leaves the reference prompt and a few rows of headroom
    // below the fixed flow window.  An estimate that lands exactly at the
    // limit is safe: if the sampled sequence is longer, the flow runner opens
    // another window with left context instead of truncating audio.
    const double limit = std::max(8.0, (double) gen_cap);
    auto cost = [](uint32_t cp) {
        // Whole-sentence CPU runs on the evaluation set produce about six
        // speech tokens per CJK character and two per Latin character.  The
        // previous values counted split-segment silence as speech and cut an
        // ordinary English sentence into six unnecessary windows.
        if (cp >= 0x2E80) return 6.0;      // CJK, kana, hangul
        if (cp < 0x80) return 1.8;         // ASCII
        return 4.0;
    };
    auto strong = [](uint32_t cp) {
        return cp == '.' || cp == '!' || cp == '?' || cp == 0x3002 || cp == 0xFF01 ||
               cp == 0xFF1F || cp == ';' || cp == 0xFF1B || cp == '\n' || cp == 0x2026;
    };
    auto weak = [](uint32_t cp) {
        return cp == ',' || cp == 0xFF0C || cp == 0x3001 || cp == ':' || cp == 0xFF1A || cp == ' ';
    };
    std::vector<std::string> out;
    size_t start = 0;
    while (start < cps.size()) {
        double used = 0.0;
        size_t last_strong = 0, last_weak = 0;
        size_t i = start;
        for (; i < cps.size(); i++) {
            used += cost(cps[i].cp);
            if (strong(cps[i].cp)) last_strong = i + 1;
            else if (weak(cps[i].cp)) last_weak = i + 1;
            if (used > limit) break;
        }
        size_t cut;
        if (i >= cps.size()) {
            cut = cps.size();
        } else if (last_strong > start) {
            cut = last_strong;
        } else if (last_weak > start) {
            cut = last_weak;
        } else {
            cut = i + 1;  // one unpunctuated clause: hard split so it progresses
        }
        size_t b = cps[start].off;
        size_t e = cut < cps.size() ? cps[cut].off : text.size();
        while (b < e && (text[b] == ' ' || text[b] == '\t' || text[b] == '\n')) b++;
        while (e > b && (text[e - 1] == ' ' || text[e - 1] == '\t' || text[e - 1] == '\n')) e--;
        if (e > b) out.push_back(text.substr(b, e - b));
        start = cut;
    }
    if (out.empty()) out.push_back(text);
    return out;
}

// Concatenate mono 24 kHz 16-bit RIFF files into one file.
static bool concat_wav_parts(const std::vector<std::string> & parts, const std::string & outpath) {
    std::vector<uint8_t> pcm;
    for (const auto & p : parts) {
        std::vector<uint8_t> wav;
        if (!read_binary_file(p, wav) || wav.size() <= 44) return false;
        pcm.insert(pcm.end(), wav.begin() + 44, wav.end());
    }
    std::vector<uint8_t> out;
    out.reserve(44 + pcm.size());
    auto put32 = [&](uint32_t v) {
        out.push_back(v & 0xFF); out.push_back((v >> 8) & 0xFF);
        out.push_back((v >> 16) & 0xFF); out.push_back((v >> 24) & 0xFF);
    };
    auto put16 = [&](uint16_t v) { out.push_back(v & 0xFF); out.push_back((v >> 8) & 0xFF); };
    const uint32_t data_size = (uint32_t) pcm.size();
    out.insert(out.end(), {'R', 'I', 'F', 'F'}); put32(36 + data_size);
    out.insert(out.end(), {'W', 'A', 'V', 'E', 'f', 'm', 't', ' '}); put32(16);
    put16(1); put16(1); put32(24000); put32(48000); put16(2); put16(16);
    out.insert(out.end(), {'d', 'a', 't', 'a'}); put32(data_size);
    out.insert(out.end(), pcm.begin(), pcm.end());
    return write_binary_file(outpath, out);
}

// ---------------------------------------------------------------------------
// AXERA engine wrapper

#ifdef COSY_TTS_HAVE_AXERA
struct npu_model {
    AX_ENGINE_HANDLE handle = nullptr;
    AX_ENGINE_IO_INFO_T * info = nullptr;
    // AX_ENGINE_CreateHandle does not copy the model blob.  The public AXERA
    // backend keeps the same bytes alive for the handle's full lifetime; doing
    // otherwise leaves the device reading freed host memory after the first
    // launch and eventually corrupts the allocator.
    std::vector<char> model_blob;
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;

    npu_model() = default;
    npu_model(const npu_model &) = delete;
    npu_model & operator=(const npu_model &) = delete;

    bool load(const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open axmodel %s\n", path.c_str()); return false; }
        model_blob.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        if (model_blob.empty() || AX_ENGINE_CreateHandle(&handle, model_blob.data(), (AX_U32) model_blob.size()) != 0 ||
            AX_ENGINE_CreateContext(handle) != 0 || AX_ENGINE_GetIOInfo(handle, &info) != 0) {
            fprintf(stderr, "failed to load axmodel %s\n", path.c_str());
            return false;
        }
        in.resize(info->nInputSize);
        out.resize(info->nOutputSize);
        auto alloc = [](AX_ENGINE_IO_BUFFER_T & b, AX_U32 size, bool pad64) {
            const AX_U32 alloc_size = pad64 ? (size + 63u) & ~63u : size;
            if (AX_SYS_MemAllocCached(&b.phyAddr, &b.pVirAddr, alloc_size, 4096, (const AX_S8 *) "cosy-tts") != 0) {
                return false;
            }
            b.nSize = alloc_size;
            memset(b.pVirAddr, 0, alloc_size);
            return true;
        };
        for (AX_U32 i = 0; i < info->nInputSize; i++) {
            if (!alloc(in[i], info->pInputs[i].nSize, false)) { fprintf(stderr, "input CMM alloc failed\n"); return false; }
        }
        for (AX_U32 i = 0; i < info->nOutputSize; i++) {
            // HiFT's 18x61441 f32 output is not a multiple of 64 bytes.  The
            // board runtime splits that final partial vector into the two
            // phase rows, duplicating channel 16 into channel 17.  Allocate a
            // 64-byte-aligned output capacity while keeping the graph's shape.
            if (!alloc(out[i], info->pOutputs[i].nSize, true)) { fprintf(stderr, "output CMM alloc failed\n"); return false; }
        }
        for (AX_U32 i = 0; i < info->nInputSize; i++) {
            const auto & m = info->pInputs[i];
            fprintf(stderr, "[npu] in  %-10s %u bytes dtype=%d layout=%d shape=[", m.pName, m.nSize,
                    (int) m.eDataType, (int) m.eLayout);
            for (int d = 0; d < m.nShapeSize; d++) fprintf(stderr, "%s%d", d ? "," : "", m.pShape[d]);
            fprintf(stderr, "] stride=[");
            if (m.pStride) for (int d = 0; d < m.nShapeSize; d++) fprintf(stderr, "%s%d", d ? "," : "", m.pStride[d]);
            fprintf(stderr, "]\n");
        }
        for (AX_U32 i = 0; i < info->nOutputSize; i++) {
            const auto & m = info->pOutputs[i];
            fprintf(stderr, "[npu] out %-10s %u bytes dtype=%d layout=%d shape=[", m.pName, m.nSize,
                    (int) m.eDataType, (int) m.eLayout);
            for (int d = 0; d < m.nShapeSize; d++) fprintf(stderr, "%s%d", d ? "," : "", m.pShape[d]);
            fprintf(stderr, "] stride=[");
            if (m.pStride) for (int d = 0; d < m.nShapeSize; d++) fprintf(stderr, "%s%d", d ? "," : "", m.pStride[d]);
            fprintf(stderr, "]\n");
        }
        return true;
    }

    int input_index(const char * name) const {
        for (AX_U32 i = 0; i < info->nInputSize; i++) if (!strcmp(info->pInputs[i].pName, name)) return (int) i;
        return -1;
    }

    size_t input_elems(const char * name) const {
        const int i = input_index(name);
        if (i < 0) return 0;
        size_t n = 1;
        for (int d = 0; d < info->pInputs[i].nShapeSize; d++) n *= (size_t) info->pInputs[i].pShape[d];
        return n;
    }

    bool set_input(const char * name, const void * data, size_t bytes) {
        const int i = input_index(name);
        if (i < 0 || bytes > in[i].nSize) {
            fprintf(stderr, "bad NPU input %s (%zu bytes, max %u)\n", name, bytes, i < 0 ? 0u : in[i].nSize);
            return false;
        }
        if (bytes < in[i].nSize) memset((char *) in[i].pVirAddr + bytes, 0, in[i].nSize - bytes);
        memcpy(in[i].pVirAddr, data, bytes);
        AX_SYS_MflushCache(in[i].phyAddr, in[i].pVirAddr, in[i].nSize);
        return true;
    }

    void * input_ptr(const char * name) {
        const int i = input_index(name);
        return i >= 0 ? in[i].pVirAddr : nullptr;
    }

    size_t input_bytes(const char * name) const {
        const int i = input_index(name);
        return i >= 0 ? (size_t) in[i].nSize : 0;
    }

    bool run() {
        // The output buffers are zeroed by the CPU at load time.  Flush those
        // dirty lines before the first device write, otherwise the first NPU
        // result can be overwritten by a late CPU cache write-back.
        for (auto & b : out) AX_SYS_MflushCache(b.phyAddr, b.pVirAddr, b.nSize);
        AX_ENGINE_IO_T io{};
        io.pInputs = in.data();
        io.nInputSize = (AX_U32) in.size();
        io.pOutputs = out.data();
        io.nOutputSize = (AX_U32) out.size();
        const int rc = AX_ENGINE_RunSync(handle, &io);
        if (rc != 0) { fprintf(stderr, "AX_ENGINE_RunSync failed 0x%x\n", rc); return false; }
        for (auto & b : out) AX_SYS_MinvalidateCache(b.phyAddr, b.pVirAddr, b.nSize);
        return true;
    }


    int output_index(const char * name) const {
        for (AX_U32 i = 0; i < info->nOutputSize; i++) if (!strcmp(info->pOutputs[i].pName, name)) return (int) i;
        return -1;
    }

    const float * output_f32(size_t expected, int idx = 0) const {
        if (idx < 0 || idx >= (int) info->nOutputSize ||
            info->pOutputs[idx].eDataType != AX_ENGINE_DT_FLOAT32 ||
            out[idx].nSize < expected * sizeof(float)) {
            fprintf(stderr, "NPU output mismatch: outputs=%u idx=%d dtype=%d bytes=%u expected=%zu\n",
                    info->nOutputSize, idx, info->pOutputs[idx].eDataType, out[idx].nSize, expected * sizeof(float));
            return nullptr;
        }
        return (const float *) out[idx].pVirAddr;
    }

    size_t output_bytes(int idx = 0) const {
        return idx >= 0 && idx < (int) info->nOutputSize ? info->pOutputs[idx].nSize : 0;
    }

    ~npu_model() {
        for (auto & b : in)  if (b.pVirAddr) AX_SYS_MemFree(b.phyAddr, b.pVirAddr);
        for (auto & b : out) if (b.pVirAddr) AX_SYS_MemFree(b.phyAddr, b.pVirAddr);
        if (handle) AX_ENGINE_DestroyHandle(handle);
    }
};
#else
struct npu_model {
    struct io_buffer { uint32_t nSize = 0; void * pVirAddr = nullptr; };
    std::vector<io_buffer> in, out;
    bool load(const std::string &) {
        fprintf(stderr, "this build has no AXERA support; rebuild with GGML_AXERA and GGML_AXERA_BSP_DIR\n");
        return false;
    }
    void * input_ptr(const char *) { return nullptr; }
    size_t input_bytes(const char *) const { return 0; }
    size_t input_elems(const char *) const { return 0; }
    int input_index(const char *) const { return -1; }
    int output_index(const char *) const { return -1; }
    bool set_input(const char *, const void *, size_t) { return false; }
    bool run() { return false; }
    const float * output_f32(size_t, int = 0) const { return nullptr; }
    size_t output_bytes(int = 0) const { return 0; }
};
#endif

// ---------------------------------------------------------------------------
// CosyVoice3 feature front ends.  Both use the same public mtmd FFT/mel helper
// as the rest of this tree; the filter/window tables are fixed constants copied
// from the official Python implementation (see ref/tables/tables.json).

// ---------------------------------------------------------------------------
// Native HiFT front/post.  The f0 predictor, SineGen2 source and both FFT
// directions are small compared with the decoder, but ONNX Runtime paid
// hundreds of milliseconds per request on them.  The weights and the
// deterministic noise/shuffle tables are extracted once at pack time.

// Read-only GGUF opened with tensor data resident in host memory.  The
// front-end aux GGUF replaces the old tables/ directory and the ONNX graphs;
// its tensors are the only weight source for the reference-audio encoders,
// the HiFT f0 front and the host feature tables.
struct gguf_asset_file {
    gguf_context * ctx = nullptr;
    ggml_context * tensors = nullptr;
    bool opened = false;

    gguf_asset_file() = default;
    gguf_asset_file(const gguf_asset_file &) = delete;
    gguf_asset_file & operator=(const gguf_asset_file &) = delete;

    bool open(const std::string & path) {
        if (opened) return true;
        gguf_init_params p{};
        p.no_alloc = false;
        p.ctx = &tensors;
        ctx = gguf_init_from_file(path.c_str(), p);
        if (!ctx || !tensors) { fprintf(stderr, "failed to load %s\n", path.c_str()); return false; }
        opened = true;
        return true;
    }

    ggml_tensor * get(const char * name) const {
        return tensors ? ggml_get_tensor(tensors, name) : nullptr;
    }

    const float * f32(const char * name, size_t expected = 0, int * ne_out = nullptr) const {
        ggml_tensor * t = get(name);
        if (!t || t->type != GGML_TYPE_F32) {
            fprintf(stderr, "front GGUF: %s missing or not F32\n", name);
            return nullptr;
        }
        if (expected && (size_t) ggml_nelements(t) != expected) {
            fprintf(stderr, "front GGUF: %s has %lld elems, expected %zu\n", name, (long long) ggml_nelements(t), expected);
            return nullptr;
        }
        if (ne_out) for (int i = 0; i < 4; i++) ne_out[i] = (int) t->ne[i];
        return (const float *) t->data;
    }

    ~gguf_asset_file() {
        if (ctx) gguf_free(ctx);
    }
};


static std::string kernel_tensor_name(const std::string & name) {
    std::string out = name;
    for (char & c : out) if (c == '.') c = '_';
    return out;
}

// The tokenizer AXModel exposes every weight as a runtime input, exactly like
// the LLM weight banks.  Convert the GGUF tensors into the layouts the tiled
// matmul primitives expect once at load time; the model file itself never
// contains weights.
static bool upload_front_weights(npu_model & m, gguf_asset_file & assets) {
    constexpr int LAYERS = 12;
    fprintf(stderr, "[front-npu] upload start\n");
    auto upload = [&](const std::string & in, const void * data, size_t bytes) {
        const int idx = m.input_index(in.c_str());
        if (idx < 0) { fprintf(stderr, "front AXModel: no input %s\n", in.c_str()); return false; }
        if (bytes != m.in[idx].nSize || !m.set_input(in.c_str(), data, bytes)) {
            fprintf(stderr, "front AXModel: %s expects %u bytes, got %zu\n", in.c_str(), m.in[idx].nSize, bytes);
            return false;
        }
        return true;
    };
    auto bf16_tensor = [&](const std::string & name) -> ggml_tensor * {
        ggml_tensor * t = ggml_get_tensor(assets.tensors, name.c_str());
        if (!t || t->type != GGML_TYPE_BF16) {
            fprintf(stderr, "front GGUF: missing or non-BF16 tensor %s\n", name.c_str());
            return nullptr;
        }
        return t;
    };

    // Non-banked tensors: the two stride-2 convolutions, the FSQ projection and
    // its constants.
    int uploaded = 0;
    for (const char * name : {"tok.conv1.weight", "tok.conv1.bias", "tok.conv2.weight", "tok.conv2.bias"}) {
        ggml_tensor * t = ggml_get_tensor(assets.tensors, name);
        if (!t) { fprintf(stderr, "front GGUF: missing %s\n", name); return false; }
        const size_t nelem = (size_t) ggml_nelements(t);
        std::string in = kernel_tensor_name(name);
        in = in.substr(0, in.size() - 7) + "_mat";
        if (std::string(name).find(".bias") != std::string::npos) in = kernel_tensor_name(name);
        std::vector<uint16_t> buf(nelem);
        if (std::string(name).find(".weight") != std::string::npos) {
            const int K = (int) t->ne[0], IC = (int) t->ne[1], OC = (int) t->ne[2];
            const uint16_t * src = (const uint16_t *) t->data;
            for (int oc = 0; oc < OC; oc++)
                for (int ic = 0; ic < IC; ic++)
                    for (int k = 0; k < K; k++)
                        buf[((size_t) oc * K + k) * IC + ic] = src[((size_t) oc * IC + ic) * K + k];
        } else {
            memcpy(buf.data(), t->data, nelem * 2);
        }
        if (!upload(in, buf.data(), buf.size() * 2)) return false;
        uploaded++;
    }
    {
        // Official graph projects 1280 -> 8.  A 32-wide matmul is the smallest
        // shape the tiling primitive accepts; rows 8..31 stay zero.
        ggml_tensor * qw = bf16_tensor("tok.qproj.weight");
        ggml_tensor * qb = bf16_tensor("tok.qproj.bias");
        if (!qw || !qb || qw->ne[0] != 8 || qw->ne[1] != 1280) {
            fprintf(stderr, "[front-npu] bad qproj: w=%p b=%p ne=(%lld,%lld)\n", (void *) qw, (void *) qb,
                    qw ? (long long) qw->ne[0] : -1, qw ? (long long) qw->ne[1] : -1);
            return false;
        }
        const uint16_t * src = (const uint16_t *) qw->data;
        std::vector<uint16_t> pad((size_t) 32 * 1280, 0);
        for (int r = 0; r < 8; r++)
            for (int c = 0; c < 1280; c++) pad[(size_t) r * 1280 + c] = src[(size_t) c * 8 + r];
        if (!upload("tok_qproj_pad", pad.data(), pad.size() * 2)) return false;
        if (!upload("tok_qproj_bias", qb->data, 8 * 2)) return false;
        uploaded += 2;
    }
    for (const char * name : {"tok.fsq.Constant_6", "tok.fsq.Constant_10", "tok.fsq.Constant_11"}) {
        ggml_tensor * t = ggml_get_tensor(assets.tensors, name);
        if (!t || t->type != GGML_TYPE_F32) { fprintf(stderr, "front GGUF: missing %s\n", name); return false; }
        if (!upload(kernel_tensor_name(name), t->data, (size_t) ggml_nelements(t) * 4)) return false;
        uploaded++;
    }

    // Layer banks: every block weight is concatenated along rows, so the
    // kernel selects a layer with one index_select and keeps a single body.
    struct bank_spec { const char * in; const char * suffix; int pad; };
    static const bank_spec banks[] = {
        {"tok_blk_ln1_weight", "ln1.weight", 0}, {"tok_blk_ln1_bias", "ln1.bias", 0},
        {"tok_blk_q_mat",      "q.mat",      0}, {"tok_blk_q_bias",   "q.bias",   0},
        {"tok_blk_k_mat",      "k.mat",      0}, {"tok_blk_v_mat",    "v.mat",    0},
        {"tok_blk_v_bias",     "v.bias",     0}, {"tok_blk_o_mat",    "o.mat",    0},
        {"tok_blk_o_bias",     "o.bias",     0}, {"tok_blk_fsmn_weight", "fsmn.weight", 32},
        {"tok_blk_ln2_weight", "ln2.weight", 0}, {"tok_blk_ln2_bias", "ln2.bias", 0},
        {"tok_blk_ff1_mat",    "ff1.mat",    0}, {"tok_blk_ff1_bias", "ff1.bias", 0},
        {"tok_blk_ff2_mat",    "ff2.mat",    0}, {"tok_blk_ff2_bias", "ff2.bias", 0},
    };
    for (const auto & spec : banks) {
        const int idx = m.input_index(spec.in);
        if (idx < 0) { fprintf(stderr, "front AXModel: no input %s\n", spec.in); return false; }
        const size_t want = m.in[idx].nSize / 2;             // BF16 elements
        const std::string suffix = spec.suffix;
        const bool is_mat = suffix.size() > 4 && suffix.compare(suffix.size() - 4, 4, ".mat") == 0;
        // Kernel-side `*.mat` tensors are the GGUF `*.weight` initializers.
        const std::string gguf_suffix = is_mat ? suffix.substr(0, suffix.size() - 3) + "weight" : suffix;
        std::vector<uint16_t> buf(want, 0);
        size_t at = 0;
        fprintf(stderr, "[front-npu] bank %s <- 12 layers, want %zu\n", spec.suffix, want);
        for (int l = 0; l < LAYERS; l++) {
            ggml_tensor * t = bf16_tensor("tok.blk" + std::to_string(l) + "." + gguf_suffix);
            if (!t) return false;
            const uint16_t * src = (const uint16_t *) t->data;
            if (is_mat) {
                // GGUF stores a linear weight with OC as the fastest dimension.
                const int OC = (int) t->ne[0], IC = (int) t->ne[1];
                if (at + (size_t) IC * OC > want) {
                    fprintf(stderr, "[front-npu] %s overflow at layer %d: %zu + %d > %zu\n",
                            spec.suffix, l, at, IC * OC, want);
                    return false;
                }
                for (int oc = 0; oc < OC; oc++)
                    for (int ic = 0; ic < IC; ic++)
                        buf[at + (size_t) oc * IC + ic] = src[(size_t) ic * OC + oc];
                at += (size_t) IC * OC;
            } else if (spec.pad) {
                const int K = (int) t->ne[0], C = (int) t->ne[2];
                if (at + (size_t) C * spec.pad > want) {
                    fprintf(stderr, "[front-npu] %s overflow at layer %d: %zu + %d > %zu\n",
                            spec.suffix, l, at, C * spec.pad, want);
                    return false;
                }
                for (int c = 0; c < C; c++)
                    for (int k = 0; k < K; k++)
                        buf[at + (size_t) c * spec.pad + k] = src[(size_t) c * K + k];
                at += (size_t) C * spec.pad;
            } else {
                const size_t n = (size_t) ggml_nelements(t);
                if (at + n > want) {
                    fprintf(stderr, "[front-npu] %s overflow at layer %d: %zu + %zu > %zu\n",
                            spec.suffix, l, at, n, want);
                    return false;
                }
                memcpy(&buf[at], src, n * 2);
                at += n;
            }
        }
        if (at != want) {
            fprintf(stderr, "front GGUF: bank %s has %zu elements, kernel wants %zu\n", spec.suffix, at, want);
            return false;
        }
        if (!upload(spec.in, buf.data(), buf.size() * 2)) return false;
        uploaded++;
    }
    fprintf(stderr, "[front-npu] uploaded %d tokenizer tensors (12-layer banks)\n", uploaded);
    return uploaded > 0;
}


// Fixed 30 s reference-audio window: 3000 Whisper mel frames -> 750 language
// model tokens.  Padding is zero (the same value the official feature
// extractor uses for silence); anything longer is truncated like the official
// front end.  All other inputs are computed here, so the AXModel only sees
// plain tensors.
static bool front_tokenize(npu_model & m, const std::vector<float> & wmel, int frames,
                           std::vector<int32_t> & tokens) {
    constexpr int WIN = 3000, TOK = 750;
    const int valid = std::min(frames, WIN);
    if (valid <= 0) { fprintf(stderr, "empty reference mel\\n"); return false; }
    std::vector<ggml_bf16_t> mel((size_t) WIN * 128, ggml_bf16_t{0});
    for (int c = 0; c < 128; c++)
        for (int t = 0; t < valid; t++)
            mel[(size_t) t * 128 + c] = ggml_fp32_to_bf16(wmel[(size_t) c * frames + t]);
    const int ntok = std::min(TOK, (valid + 3) / 4);
    std::vector<float> mask((size_t) TOK, 0.0f), bias((size_t) TOK, -1e9f);
    for (int i = 0; i < ntok; i++) { mask[i] = 1.0f; bias[i] = 0.0f; }
    std::vector<ggml_bf16_t> cosv((size_t) TOK * 64), sinv((size_t) TOK * 64);
    for (int p = 0; p < TOK; p++) {
        for (int j = 0; j < 32; j++) {
            const double inv = 1.0 / std::pow(10000.0, (double) (2 * j) / 64.0);
            const double ang = (double) p * inv;
            const ggml_bf16_t cv = ggml_fp32_to_bf16((float) std::cos(ang));
            const ggml_bf16_t sv = ggml_fp32_to_bf16((float) std::sin(ang));
            cosv[(size_t) p * 64 + j] = cv;      cosv[(size_t) p * 64 + 32 + j] = cv;
            sinv[(size_t) p * 64 + j] = sv;      sinv[(size_t) p * 64 + 32 + j] = sv;
        }
    }
    std::vector<int32_t> tok_out((size_t) TOK, 0);
    std::vector<ggml_bf16_t> hidden((size_t) TOK * 1280, ggml_bf16_t{0});
    if (!m.set_input("mel", mel.data(), mel.size() * 2) ||
        !m.set_input("mask", mask.data(), mask.size() * 4) ||
        !m.set_input("attn_bias", bias.data(), bias.size() * 4) ||
        !m.set_input("cos", cosv.data(), cosv.size() * 2) ||
        !m.set_input("sin", sinv.data(), sinv.size() * 2) || !m.run()) {
        return false;
    }
    const int32_t * result = nullptr;
    const int oi = m.output_index("tokens");
    if (oi >= 0 && m.output_bytes(oi) >= tok_out.size() * 4) {
        result = (const int32_t *) m.out[oi].pVirAddr;
    } else {
        result = (const int32_t *) m.input_ptr("tokens");
    }
    if (!result) { fprintf(stderr, "front AXModel produced no token output\\n"); return false; }
    tokens.assign(result, result + ntok);
    return true;
}

struct hift_native {
    struct layer {
        int out_c, in_c, k;
        const float * w, * b;
        // [in_channel][kernel][out_channel].  The source table is
        // out-channel-major, which makes the 16-wide SIMD accumulator walk
        // weights with a stride.  This load-time transpose keeps each output
        // block contiguous and does not change the convolution result.
        std::vector<float> wt;
    };
    std::vector<layer> layers;
    const float * classifier_w = nullptr, * classifier_b = nullptr;
    const float * linear_w = nullptr, * linear_b = nullptr;
    const float * rand_ini = nullptr;
    const float * noise = nullptr;
    size_t noise_samples = 0;

    bool load(gguf_asset_file & g) {
        char name[64];
        layers.clear();
        for (int i = 0; ; i++) {
            snprintf(name, sizeof(name), "hift.front.l%d.w", i);
            ggml_tensor * tw = g.get(name);
            if (!tw) break;
            snprintf(name, sizeof(name), "hift.front.l%d.b", i);
            ggml_tensor * tb = g.get(name);
            if (!tb || tw->type != GGML_TYPE_F32 || tb->type != GGML_TYPE_F32) {
                fprintf(stderr, "front GGUF: bad HiFT f0 layer %d\n", i);
                return false;
            }
            layer l;
            l.k = (int) tw->ne[0];
            l.in_c = (int) tw->ne[1];
            l.out_c = (int) tw->ne[2];
            l.w = (const float *) tw->data;
            l.b = (const float *) tb->data;
            const size_t wb = (size_t) l.out_c * l.in_c * l.k;
            l.wt.resize(wb);
            for (int ic = 0; ic < l.in_c; ic++) {
                for (int k = 0; k < l.k; k++) {
                    for (int oc = 0; oc < l.out_c; oc++) {
                        l.wt[((size_t) ic * l.k + k) * l.out_c + oc] =
                            l.w[((size_t) oc * l.in_c + ic) * l.k + k];
                    }
                }
            }
            layers.push_back(l);
        }
        if (layers.empty()) { fprintf(stderr, "front GGUF: no HiFT f0 layers\n"); return false; }
        classifier_w = g.f32("hift.front.classifier.w", 512);
        classifier_b = g.f32("hift.front.classifier.b", 1);
        linear_w = g.f32("hift.front.linear.w", 9);
        linear_b = g.f32("hift.front.linear.b", 1);
        rand_ini = g.f32("hift.front.rand_ini", 9);
        ggml_tensor * tn = g.get("hift.front.noise");
        if (!classifier_w || !classifier_b || !linear_w || !linear_b || !rand_ini ||
            !tn || tn->type != GGML_TYPE_F32 || tn->ne[0] != 9) {
            fprintf(stderr, "front GGUF: missing HiFT source tensors\n");
            return false;
        }
        noise_samples = (size_t) tn->ne[1];
        noise = (const float *) tn->data;
        fprintf(stderr, "[hift-native] loaded %zu layers, noise_samples=%zu from GGUF\n", layers.size(), noise_samples);
        return true;
    }
};

template <typename Fn>
static void parallel_ranges(int n, int max_threads, Fn && fn);

// Input and output are time-major [T][channels].  A scalar time-major loop
// reloads the 3.3 MB f0 predictor for every frame.  Keep a short time tile in
// registers so each block of weights is reused across eight frames while the
// inner 16-output FMA stays contiguous.
static void conv_causal_parallel(const float * x, int T, const hift_native::layer & l,
                                 bool right, std::vector<float> & y) {
    constexpr int BLK = 32;
    constexpr int BT = 4;
    y.resize((size_t) l.out_c * T);
    const int blocks = l.out_c / BLK;
    parallel_ranges(blocks, 16, [&](int b0, int b1) {
        for (int blk = b0; blk < b1; blk++) {
            const int oc0 = blk * BLK;
            const int tend = (T / BT) * BT;
            for (int t0 = 0; t0 < tend; t0 += BT) {
                float acc[BT][BLK];
                #pragma GCC unroll 4
                for (int t = 0; t < BT; t++)
                    #pragma GCC ivdep
                    for (int v = 0; v < BLK; v++) acc[t][v] = l.b[oc0 + v];
                for (int ic = 0; ic < l.in_c; ic++) {
                    for (int k = 0; k < l.k; k++) {
                        const int off = right ? k : k - (l.k - 1);
                        const float * __restrict w = l.wt.data() +
                            ((size_t) ic * l.k + k) * l.out_c + oc0;
                        #pragma GCC unroll 4
                        for (int t = 0; t < BT; t++) {
                            const int st = t0 + t + off;
                            if (st < 0 || st >= T) continue;
                            const float xv = x[(size_t) st * l.in_c + ic];
                            #pragma GCC ivdep
                            for (int v = 0; v < BLK; v++) acc[t][v] += xv * w[v];
                        }
                    }
                }
                #pragma GCC unroll 4
                for (int t = 0; t < BT; t++) {
                    float * __restrict dst = y.data() + (size_t) (t0 + t) * l.out_c + oc0;
                    #pragma GCC ivdep
                    for (int v = 0; v < BLK; v++) {
                        const float z = acc[t][v];
                        dst[v] = z < 0.0f ? std::expm1f(z) : z;
                    }
                }
            }
            for (int t = tend; t < T; t++) {
                float acc[BLK];
                for (int v = 0; v < BLK; v++) acc[v] = l.b[oc0 + v];
                for (int ic = 0; ic < l.in_c; ic++) for (int k = 0; k < l.k; k++) {
                    const int off = right ? k : k - (l.k - 1);
                    const int st = t + off;
                    if (st < 0 || st >= T) continue;
                    const float xv = x[(size_t) st * l.in_c + ic];
                    const float * __restrict w = l.wt.data() +
                        ((size_t) ic * l.k + k) * l.out_c + oc0;
                    #pragma GCC ivdep
                    for (int v = 0; v < BLK; v++) acc[v] += xv * w[v];
                }
                float * __restrict dst = y.data() + (size_t) t * l.out_c + oc0;
                for (int v = 0; v < BLK; v++) {
                    const float z = acc[v];
                    dst[v] = z < 0.0f ? std::expm1f(z) : z;
                }
            }
        }
    });
    // The CosyVoice3 f0 predictor has five 512-output layers.  Keep the tail
    // path for completeness, but never silently drop a non-multiple model.
    if (l.out_c % BLK != 0) {
        fprintf(stderr, "hift conv: unsupported output width %d\n", l.out_c);
        std::abort();
    }
}

static void hift_front_native(const hift_native & h, const float * mel, int T,
                              std::vector<float> & s_stft, int & nf,
                              const double * phase0 = nullptr, double * phase_end = nullptr,
                              size_t sample_offset = 0, double * phase_frames = nullptr,
                              const float * f0_in = nullptr) {
    constexpr int HARM = 9, UPS = 480, SR = 24000, HOP = 4, KB = 9;
    const auto clock_now = [] { return std::chrono::steady_clock::now(); };
    const auto elapsed = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0 = clock_now();
    std::vector<float> f0;
    if (f0_in) {
        f0.assign(f0_in, f0_in + T);
    } else {
        std::vector<float> a((size_t) T * 80), b;
        for (int c = 0; c < 80; c++) {
            for (int t = 0; t < T; t++) a[(size_t) t * 80 + c] = mel[(size_t) c * T + t];
        }
        for (int li = 0; li < (int) h.layers.size(); li++) {
            conv_causal_parallel(a.data(), T, h.layers[li], li == 0, b);
            a.swap(b);
        }
        const int C = h.layers.back().out_c;
        f0.resize(T);
        for (int t = 0; t < T; t++) {
            double sum = h.classifier_b[0];
            for (int c = 0; c < C; c++) sum += (double) a[(size_t) t * C + c] * h.classifier_w[c];
            f0[t] = std::fabs((float) sum);
        }
    }
    const auto t1 = clock_now();

    const int L = T * UPS;
    std::vector<float> rad((size_t) HARM * T);
    for (int hh = 0; hh < HARM; hh++) {
        for (int j = 0; j < T; j++) {
            // Linear downsampling of the nearest-upsampled f0 evaluates both
            // half-sample endpoints inside the same mel frame.  rand_ini was
            // added to the first waveform sample before downsampling, so the
            // first mel frame receives half of it; all later frames are exact.
            float v = std::fmod(f0[j] * (float) (hh + 1) / (float) SR, 1.0f);
            rad[(size_t) hh * T + j] = v;
        }
    }
    std::vector<double> phase((size_t) HARM * T);
    for (int hh = 0; hh < HARM; hh++) {
        double c = phase0 ? phase0[hh] : 0.0;
        for (int j = 0; j < T; j++) {
            c += rad[(size_t) hh * T + j];
            phase[(size_t) hh * T + j] = c;
            if (phase_frames) phase_frames[(size_t) j * HARM + hh] = c;
        }
        if (phase_end) phase_end[hh] = c;
    }
    // The harmonic phase only changes once per mel frame, so evaluate the
    // nine sine waves per frame instead of once per upsampled sample.
    std::vector<float> harm_sin((size_t) HARM * T);
    for (int hh = 0; hh < HARM; hh++) {
        for (int j = 0; j < T; j++) {
            harm_sin[(size_t) hh * T + j] =
                std::sin((float) phase[(size_t) hh * T + j] * (float) UPS * 2.0f * (float) M_PI) * 0.1f;
        }
    }
    std::vector<float> source(L);
    parallel_ranges(L, 16, [&](int fa, int fb) {
        for (int f = fa; f < fb; f++) {
            const int j = f / UPS;
            const bool uv = (f0[f / UPS] > 10.0f);
            const float noise_amp = uv ? 0.003f : (0.1f / 3.0f);
            double sum = h.linear_b[0];
            for (int hh = 0; hh < HARM; hh++) {
                const float sine = harm_sin[(size_t) hh * T + j];
                const size_t noise_idx = sample_offset + (size_t) f;
                const float noise = noise_idx < h.noise_samples ? h.noise[noise_idx * 9 + hh] : 0.0f;
                const float sw = (uv ? sine : 0.0f) + noise_amp * noise;
                sum += sw * h.linear_w[hh];
            }
            source[f] = std::tanh((float) sum);
        }
    });
    const auto t2 = clock_now();

    std::vector<float> pad((size_t) L + 16);
    for (int i = 0; i < 8; i++) pad[i] = source[8 - i];
    for (int i = 0; i < L; i++) pad[8 + i] = source[i];
    for (int i = 0; i < 8; i++) pad[8 + L + i] = source[L - 2 - i];
    nf = L / HOP + 1;
    s_stft.assign((size_t) 18 * nf, 0.0f);
    float win[16], cs[16][9], sn[16][9];
    for (int n = 0; n < 16; n++) {
        win[n] = 0.5f - 0.5f * std::cos(2.0f * (float) M_PI * n / 16.0f);
        for (int k = 0; k < 9; k++) {
            const double ang = 2.0 * M_PI * n * k / 16.0;
            cs[n][k] = (float) std::cos(ang);
            sn[n][k] = (float) std::sin(ang);
        }
    }
    parallel_ranges(nf, 16, [&](int fa, int fb) {
        for (int f = fa; f < fb; f++) {
            float v[16];
            for (int n = 0; n < 16; n++) v[n] = pad[(size_t) f * HOP + n] * win[n];
            for (int k = 0; k < KB; k++) {
                double re = 0.0, im = 0.0;
                for (int n = 0; n < 16; n++) {
                    re += (double) v[n] * cs[n][k];
                    im -= (double) v[n] * sn[n][k];
                }
                s_stft[(size_t) k * nf + f] = (float) re;
                s_stft[(size_t) (KB + k) * nf + f] = (float) im;
            }
        }
    });
    if (getenv("COSY_HIFT_PROFILE")) {
        const auto t3 = clock_now();
        fprintf(stderr, "[hift-front] T=%d conv=%.4fs source=%.4fs stft=%.4fs total=%.4fs\n",
                T, elapsed(t0, t1), elapsed(t1, t2), elapsed(t2, t3), elapsed(t0, t3));
    }
}

static void hift_post_native(const float * mp, int nf, std::vector<float> & audio, int stride = 0) {
    if (stride <= 0) stride = nf;
    constexpr int KB = 9, NFFT = 16, HOP = 4;
    const int full = (nf - 1) * HOP + NFFT;
    std::vector<float> ola((size_t) full, 0.0f), env((size_t) full, 0.0f);
    float win[16], cs[16][9], sn[16][9];
    float wc[9], ws[9];
    for (int n = 0; n < 16; n++) {
        win[n] = 0.5f - 0.5f * std::cos(2.0f * (float) M_PI * n / 16.0f);
        for (int k = 0; k < 9; k++) {
            const double ang = 2.0 * M_PI * n * k / 16.0;
            cs[n][k] = (float) std::cos(ang);
            sn[n][k] = (float) std::sin(ang);
        }
    }
    for (int k = 0; k < 9; k++) {
        wc[k] = (k == 0 || k == 8) ? 1.0f / 16.0f : 2.0f / 16.0f;
        ws[k] = (k == 0 || k == 8) ? 0.0f : 2.0f / 16.0f;
    }
    // The inverse transform is independent per frame; only the overlap-add
    // tail crosses a split.  Give each worker a private accumulator and merge
    // the ranges afterwards so long utterances use the whole CPU.
    const int nparts = std::min(8, std::max(1, nf / 1024));
    std::vector<std::vector<float>> part_ola((size_t) nparts), part_env((size_t) nparts);
    std::vector<int> part_base((size_t) nparts);
    auto post_work = [&](int pi) {
        const int fs = nf * pi / nparts;
        const int fe = nf * (pi + 1) / nparts;
        const int base = fs * HOP;
        const int len = (fe - fs) * HOP + 16;
        auto & lo = part_ola[(size_t) pi];
        auto & le = part_env[(size_t) pi];
        lo.assign((size_t) len, 0.0f);
        le.assign((size_t) len, 0.0f);
        part_base[(size_t) pi] = base;
        for (int f = fs; f < fe; f++) {
            float re[9], im[9];
            for (int k = 0; k < 9; k++) {
                const float mag = std::min(mp[(size_t) k * stride + f], 100.0f);
                const float ph = mp[(size_t) (KB + k) * stride + f];
                const float sn_v = std::sin(ph);
                const float cs_v = std::cos(ph);
                re[k] = mag * cs_v;
                im[k] = mag * sn_v;
            }
            for (int n = 0; n < 16; n++) {
                float v = 0.0f;
                for (int k = 0; k < 9; k++) v += re[k] * cs[n][k] * wc[k] - im[k] * sn[n][k] * ws[k];
                const int idx = f * HOP + n - base;
                lo[(size_t) idx] += v * win[n];
                le[(size_t) idx] += win[n] * win[n];
            }
        }
    };
    if (nparts <= 1) {
        post_work(0);
    } else {
        std::vector<std::thread> workers;
        for (int i = 1; i < nparts; i++) workers.emplace_back(post_work, i);
        post_work(0);
        for (auto & w : workers) w.join();
    }
    for (int pi = 0; pi < nparts; pi++) {
        const int base = part_base[(size_t) pi];
        const int len = (int) part_ola[(size_t) pi].size();
        for (int i = 0; i < len; i++) {
            ola[(size_t) base + i] += part_ola[(size_t) pi][(size_t) i];
            env[(size_t) base + i] += part_env[(size_t) pi][(size_t) i];
        }
    }
    audio.resize((size_t) (nf - 1) * HOP);
    for (size_t i = 0; i < audio.size(); i++) {
        const float v = ola[i + 8] / (env[i + 8] + 1e-8f);
        audio[i] = std::max(-0.99f, std::min(0.99f, v));
    }
}


// ---------------------------------------------------------------------------
// Campplus speaker encoder, ported 1:1 from the official ONNX graph.  The
// network is small (about 6M parameters, 192-dim output) and only runs when
// the reference recording changes, so it stays on the host in plain C++
// instead of taking an NPU launch.  Weights are F32 tensors in the front GGUF.

struct campplus_net {
    const gguf_asset_file * assets = nullptr;
    mutable double acc_conv = 0, acc_bn = 0, acc_pool = 0, acc_copy = 0, acc_mac = 0;

    const ggml_tensor * t(const std::string & name) const {
        return assets ? assets->get(name.c_str()) : nullptr;
    }
    const float * w(const std::string & name) const {
        const ggml_tensor * x = t(name);
        return x && x->type == GGML_TYPE_F32 ? (const float *) x->data : nullptr;
    }
    int ne(const std::string & name, int d) const {
        const ggml_tensor * x = t(name);
        return x ? (int) x->ne[d] : 0;
    }

    bool load(gguf_asset_file & g) {
        assets = &g;
        if (!w("cam.head.conv1.weight") || !w("cam.xvector.dense.linear.weight") ||
            !w("cam.xvector.dense.nonlinear.batchnorm.mean")) {
            fprintf(stderr, "front GGUF: Campplus tensors missing\n");
            return false;
        }
        return true;
    }

    // F.conv2d on [IC][IH][IW] with symmetric padding.
    void conv2d(const std::vector<float> & x, int ic, int ih, int iw, const std::string & name,
                int sh, int sw, int ph, int pw, std::vector<float> & y, int & oc, int & oh, int & ow) const {
        const auto tc0 = std::chrono::steady_clock::now();
        const float * wt = w(name + ".weight");
        const float * bt = w(name + ".bias");
        // GGUF Conv2d weight [OC][IC][KH][KW] has KW as the fastest dimension.
        const int kw = ne(name + ".weight", 0), kh = ne(name + ".weight", 1);
        oc = ne(name + ".weight", 3);
        oh = (ih + 2 * ph - kh) / sh + 1;
        ow = (iw + 2 * pw - kw) / sw + 1;
        y.assign((size_t) oc * oh * ow, 0.0f);
        acc_mac += (double) oc * oh * ow * ic * kh * kw;
        // Block eight output channels and eight x positions per accumulator
        // tile.  The old channel-major loop streamed the whole input plane
        // once per output channel, which is four gigabytes of reads for the
        // head alone.  Reusing one input vector across eight output channels
        // cuts that by eight and keeps the tile in registers.
        constexpr int OB = 8, XB = 8;
        const int ogroups = (oc + OB - 1) / OB;
        parallel_ranges(ogroups * oh, 16, [&](int q0, int q1) {
            for (int q = q0; q < q1; q++) {
                const int o0 = (q / oh) * OB;
                const int yy = q % oh;
                const int ob = std::min(OB, oc - o0);
                const int iy0 = yy * sh - ph;
                for (int x0 = 0; x0 < ow; x0 += XB) {
                    const int nb = std::min(XB, ow - x0);
                    float acc[OB][XB];
                    #pragma GCC unroll 8
                    for (int i = 0; i < OB; i++) {
                        const float b = bt ? bt[o0 + std::min(i, ob - 1)] : 0.0f;
                        #pragma GCC unroll 8
                        for (int v = 0; v < XB; v++) acc[i][v] = b;
                    }
                    for (int ky = 0; ky < kh; ky++) {
                        const int iy = iy0 + ky;
                        if (iy < 0 || iy >= ih) continue;
                        for (int kx = 0; kx < kw; kx++) {
                            const int base = x0 + kx - pw;
                            const int lo = std::max(0, -base);
                            const int hi = std::min(nb, iw - base);
                            if (hi <= lo) continue;
                            for (int c = 0; c < ic; c++) {
                                float wv[OB];
                                #pragma GCC unroll 8
                                for (int i = 0; i < OB; i++)
                                    wv[i] = wt[(((size_t) (o0 + std::min(i, ob - 1)) * ic + c) * kh + ky) * kw + kx];
                                const float * srow = x.data() + ((size_t) c * ih + iy) * iw + base;
                                if (lo == 0 && hi == XB) {
                                    float xv[XB];
                                    #pragma GCC unroll 8
                                    for (int v = 0; v < XB; v++) xv[v] = srow[v];
                                    #pragma GCC unroll 8
                                    for (int i = 0; i < OB; i++) {
                                        #pragma GCC unroll 8
                                        for (int v = 0; v < XB; v++) acc[i][v] += wv[i] * xv[v];
                                    }
                                } else {
                                    for (int v = lo; v < hi; v++)
                                        for (int i = 0; i < OB; i++) acc[i][v] += wv[i] * srow[v];
                                }
                            }
                        }
                    }
                    #pragma GCC unroll 8
                    for (int i = 0; i < ob; i++) {
                        float * drow = y.data() + (size_t) (o0 + i) * oh * ow + (size_t) yy * ow + x0;
                        #pragma GCC unroll 8
                        for (int v = 0; v < nb; v++) drow[v] = acc[i][v];
                    }
                }
            }
        });
        acc_conv += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
    }

    // C[oc][t] += A[oc][ic] * B[ic][t].  The activation matrix B is
    // time-major, so transpose A into [ic][oc] and run an 8x8 register tile:
    // 16 vector registers hold all 64 accumulators, two hold eight A values
    // and two hold eight B values.  The previous 4x8 tile kept its
    // accumulators in stack memory and reloaded them on every FMA.
    void gemm_acc(const float * A, int a_o, int a_c, const float * B, int b_c,
                  float * C, int c_row, int M, int N, int K) const {
        constexpr int MB = 8, NB = 8;
        const int mp = (M + MB - 1) / MB * MB;
        const int np = (N + 15) / 16 * 16;   // 64-byte rows: no false sharing
        std::vector<float> at((size_t) K * mp, 0.0f);
        std::vector<float> bp((size_t) K * np, 0.0f);
        for (int c = 0; c < K; c++) {
            float * dst = at.data() + (size_t) c * mp;
            const float * src = A + (size_t) c * a_c;
            for (int i = 0; i < M; i++) dst[i] = src[(size_t) i * a_o];
            std::memcpy(bp.data() + (size_t) c * np, B + (size_t) c * b_c,
                        (size_t) N * sizeof(float));
        }
        std::vector<float> cp((size_t) mp * np, 0.0f);
        const int oblks = mp / MB, nblks = np / NB;
        parallel_ranges(oblks, 16, [&](int b0, int b1) {
            for (int ob = b0; ob < b1; ob++) {
                const int o = ob * MB;
                for (int tb = 0; tb < nblks; tb++) {
                    const int t = tb * NB;
                    float acc[MB][NB] = {};
                    for (int c = 0; c < K; c++) {
                        const float * ap = at.data() + (size_t) c * mp + o;
                        const float * bq = bp.data() + (size_t) c * np + t;
                        float av[MB], bv[NB];
                        #pragma GCC unroll 8
                        for (int i = 0; i < MB; i++) av[i] = ap[i];
                        #pragma GCC unroll 8
                        for (int j = 0; j < NB; j++) bv[j] = bq[j];
                        #pragma GCC unroll 8
                        for (int i = 0; i < MB; i++) {
                            #pragma GCC unroll 8
                            for (int j = 0; j < NB; j++) acc[i][j] += av[i] * bv[j];
                        }
                    }
                    #pragma GCC unroll 8
                    for (int i = 0; i < MB; i++) {
                        float * dst = cp.data() + (size_t) (o + i) * np + t;
                        #pragma GCC unroll 8
                        for (int j = 0; j < NB; j++) dst[j] = acc[i][j];
                    }
                }
            }
        });
        for (int i = 0; i < M; i++) {
            const float * src = cp.data() + (size_t) i * np;
            float * dst = C + (size_t) i * c_row;
            for (int t = 0; t < N; t++) dst[t] += src[t];
        }
    }

    // F.conv1d on [IC][T]; the kernel has no groups.
    void conv1d(const std::vector<float> & x, int ic, int T, const std::string & name,
                int stride, int pad, int dil, std::vector<float> & y) const {
        const auto tc0 = std::chrono::steady_clock::now();
        const float * wt = w(name + ".weight");
        const float * bt = w(name + ".bias");
        const int k = ne(name + ".weight", 0);
        const int oc = ne(name + ".weight", 2);
        const int ot = (T + 2 * pad - dil * (k - 1) - 1) / stride + 1;
        y.assign((size_t) oc * ot, 0.0f);
        acc_mac += (double) oc * ot * ic * k;
        if (stride == 1) {
            for (int o = 0; o < oc; o++) {
                float * dst = y.data() + (size_t) o * ot;
                const float b = bt ? bt[o] : 0.0f;
                for (int q = 0; q < ot; q++) dst[q] = b;
            }
            for (int kk = 0; kk < k; kk++) {
                const int shift = kk * dil - pad;
                const int q0 = std::max(0, -shift);
                const int q1 = std::min(ot, T - shift);
                if (q1 <= q0) continue;
                gemm_acc(wt + kk, ic * k, k, x.data() + q0 + shift, T,
                         y.data() + q0, ot, oc, q1 - q0, ic);
            }
            acc_conv += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
            return;
        }
        parallel_ranges(oc, 16, [&](int c0, int c1) {
            for (int o = c0; o < c1; o++) {
                float * dst = y.data() + (size_t) o * ot;
                const float b = bt ? bt[o] : 0.0f;
                for (int q = 0; q < ot; q++) dst[q] = b;
                for (int kk = 0; kk < k; kk++) {
                    const int shift = kk * dil - pad;
                    for (int c = 0; c < ic; c++) {
                        const float wv = wt[((size_t) o * ic + c) * k + kk];
                        if (wv == 0.0f) continue;
                        const float * src = x.data() + (size_t) c * T;
                        int q0 = 0, q1 = ot;
                        if (stride == 1) {
                            q0 = std::max(0, -shift);
                            q1 = std::min(ot, T - shift);
                            #pragma GCC ivdep
                            for (int q = q0; q < q1; q++) dst[q] += wv * src[q + shift];
                        } else {
                            for (int q = 0; q < ot; q++) {
                                const int pos = q * stride + shift;
                                if (pos >= 0 && pos < T) dst[q] += wv * src[pos];
                            }
                        }
                    }
                }
            }
        });
        acc_conv += std::chrono::duration<double>(std::chrono::steady_clock::now() - tc0).count();
    }

    void bn(std::vector<float> & x, int C, int T, const std::string & name) const {
        const auto tb0 = std::chrono::steady_clock::now();
        const float * mean = w(name + ".mean");
        const float * var = w(name + ".var");
        const float * gamma = w(name + ".gamma");
        const float * beta = w(name + ".beta");
        const float eps = 1e-5f;
        parallel_ranges(C, 16, [&](int c0, int c1) {
            for (int c = c0; c < c1; c++) {
                const float scale = (gamma ? gamma[c] : 1.0f) / std::sqrt(var[c] + eps);
                const float shift = (beta ? beta[c] : 0.0f) - mean[c] * scale;
                float * row = x.data() + (size_t) c * T;
                for (int q = 0; q < T; q++) row[q] = row[q] * scale + shift;
            }
        });
        acc_bn += std::chrono::duration<double>(std::chrono::steady_clock::now() - tb0).count();
    }

    static void relu_inplace(std::vector<float> & x) {
        for (auto & v : x) v = std::max(v, 0.0f);
    }

    static std::vector<float> seg_pool(const std::vector<float> & x, int C, int T, double * acc = nullptr) {
        const auto tp0 = std::chrono::steady_clock::now();
        std::vector<float> out(x.size());
        const int nseg = (T + 99) / 100;
        for (int c = 0; c < C; c++) {
            for (int s = 0; s < nseg; s++) {
                const int a = s * 100, b = std::min(T, a + 100);
                double sum = 0.0;
                for (int q = a; q < b; q++) sum += x[(size_t) c * T + q];
                const float m = (float) (sum / (b - a));
                for (int q = a; q < b; q++) out[(size_t) c * T + q] = m;
            }
        }
        if (acc) *acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - tp0).count();
        return out;
    }

    std::vector<float> block(const std::vector<float> & xin, int cin, int T, int n, const char * tag) const {
        std::vector<float> x = xin;
        int c = cin;
        const int layers = n == 1 ? 12 : (n == 2 ? 24 : 16);
        const int pad = n == 1 ? 1 : 2;
        const int dil = n == 1 ? 1 : 2;
        std::vector<float> b;
        for (int i = 1; i <= layers; i++) {
            const std::string p = std::string("cam.xvector.block") + tag + ".tdnnd" + std::to_string(i);
            // h = relu(bn(x)); h = relu(conv1(h, linear1))
            b = x; bn(b, c, T, p + ".nonlinear1.batchnorm"); relu_inplace(b);
            std::vector<float> h; conv1d(b, c, T, p + ".linear1", 1, 0, 1, h);
            relu_inplace(h);
            std::vector<float> y; conv1d(h, 128, T, p + ".cam_layer.linear_local", 1, pad, dil, y);
            // context = mean over time + segment means
            std::vector<float> ctx((size_t) 128 * T);
            for (int ch = 0; ch < 128; ch++) {
                double sum = 0.0;
                for (int q = 0; q < T; q++) sum += h[(size_t) ch * T + q];
                const float m = (float) (sum / T);
                for (int q = 0; q < T; q++) ctx[(size_t) ch * T + q] = m;
            }
            std::vector<float> sp = seg_pool(h, 128, T, &acc_pool);
            for (size_t q = 0; q < ctx.size(); q++) ctx[q] += sp[q];
            std::vector<float> c1; conv1d(ctx, 128, T, p + ".cam_layer.linear1", 1, 0, 1, c1);
            relu_inplace(c1);
            std::vector<float> m; conv1d(c1, 64, T, p + ".cam_layer.linear2", 1, 0, 1, m);
            for (auto & v : m) v = 1.0f / (1.0f + std::exp(-v));
            std::vector<float> nx((size_t) (c + 32) * T);
            std::memcpy(nx.data(), x.data(), (size_t) c * T * sizeof(float));
            for (size_t q = 0; q < (size_t) 32 * T; q++) nx[(size_t) c * T + q] = y[q] * m[q];
            x.swap(nx);
            c += 32;
        }
        return x;
    }

    std::vector<float> forward(const std::vector<float> & feat, int T, const char * tag,
                               std::vector<double> * timings = nullptr) const {
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto el = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
        auto t0 = now();
        // [T][80] -> [1][1][80][T]
        std::vector<float> x((size_t) T * 80);
        for (int c = 0; c < 80; c++) for (int q = 0; q < T; q++) x[(size_t) c * T + q] = feat[(size_t) q * 80 + c];
        int ic = 1, ih = 80, iw = T, oc, oh, ow;
        std::vector<float> y;
        conv2d(x, ic, ih, iw, "cam.head.conv1", 1, 1, 1, 1, y, oc, oh, ow); relu_inplace(y);
        for (int blk = 0; blk < 2; blk++) {
            for (int i = 0; i < 2; i++) {
                const std::string p = std::string("cam.head.layer") + (blk ? "2" : "1") + ".layer" +
                                      (blk ? "2" : "1") + "." + std::to_string(i);
                const int sh = i == 0 ? 2 : 1, sw = 1;
                std::vector<float> h, y2, sc;
                int hc, hh, hw;
                conv2d(y, oc, oh, ow, p + ".conv1", sh, sw, 1, 1, h, hc, hh, hw); relu_inplace(h);
                conv2d(h, hc, hh, hw, p + ".conv2", 1, 1, 1, 1, y2, oc, oh, ow);
                if (i == 0) {
                    conv2d(y, oc, ih, iw, p + ".shortcut.shortcut.0", sh, sw, 0, 0, sc, oc, oh, ow);
                } else {
                    sc = y;
                }
                for (size_t q = 0; q < y2.size(); q++) y2[q] += sc[q];
                relu_inplace(y2);
                y.swap(y2); ih = oh; iw = ow;
            }
        }
        {
            std::vector<float> y2;
            conv2d(y, oc, oh, ow, "cam.head.conv2", 2, 1, 1, 1, y2, oc, oh, ow); relu_inplace(y2);
            ih = oh; iw = ow; y.swap(y2);
        }
        // [1][C][H][W] -> [1][C*H][W]
        const int ch = (int) (y.size() / iw);
        std::vector<float> h;
        conv1d(y, ch, iw, "cam.xvector.tdnn.linear", 2, 2, 1, h);
        auto t1 = now();
        int c = ne("cam.xvector.tdnn.linear.weight", 2);
        int tt = (int) (h.size() / c);
        for (int n = 1; n <= 3; n++) {
            h = block(h, c, tt, n, std::to_string(n).c_str());
            c += (n == 1 ? 12 : (n == 2 ? 24 : 16)) * 32;
            const std::string tr = "cam.xvector.transit" + std::to_string(n);
            bn(h, c, tt, tr + ".nonlinear.batchnorm"); relu_inplace(h);
            std::vector<float> h2; conv1d(h, c, tt, tr + ".linear", 1, 0, 1, h2);
            h.swap(h2);
            c = ne(tr + ".linear.weight", 2);
        }
        relu_inplace(h);
        auto t2 = now();
        std::vector<float> mu(c), sd(c);
        for (int ch2 = 0; ch2 < c; ch2++) {
            const float * row = h.data() + (size_t) ch2 * tt;
            double sum = 0.0;
            for (int q = 0; q < tt; q++) sum += row[q];
            const float m = (float) (sum / tt);
            double sq = 0.0;
            for (int q = 0; q < tt; q++) { const double d = row[q] - m; sq += d * d; }
            mu[ch2] = m;
            sd[ch2] = (float) std::sqrt(sq / std::max(1, tt - 1));
        }
        std::vector<float> st((size_t) 2 * c);
        std::memcpy(st.data(), mu.data(), (size_t) c * sizeof(float));
        std::memcpy(st.data() + c, sd.data(), (size_t) c * sizeof(float));
        std::vector<float> emb;
        conv1d(st, 2 * c, 1, "cam.xvector.dense.linear", 1, 0, 1, emb);
        bn(emb, 192, 1, "cam.xvector.dense.nonlinear.batchnorm");
        auto t3 = now();
        if (timings) { timings->push_back(el(t0, t1)); timings->push_back(el(t1, t2)); timings->push_back(el(t2, t3)); }
        if (getenv("COSY_PROFILE")) {
            fprintf(stderr, "[campplus-detail] conv=%.3f bn=%.3f pool=%.3f mac=%.1fM\n", acc_conv, acc_bn, acc_pool, acc_mac / 1e6);
        }
        return emb;
    }
};

struct feature_tables {
    std::vector<float> whisper_mel;    // 128 x 201
    std::vector<float> kaldi_mel;      // 80 x 257
    std::vector<float> whisper_hann;   // 400
    std::vector<float> kaldi_povey;    // 512
    std::vector<float> flow_noise;     // 1 x 80 x 15000
    std::vector<float> flow_mel;       // 80 x 961
    std::vector<float> flow_hann;      // 1920
    std::vector<float> flow_affine;    // 80 gains + 80 biases
};

static bool load_tables_gguf(gguf_asset_file & g, feature_tables & t) {
    auto cp = [&](const char * name, size_t n, std::vector<float> & dst) {
        const float * p = g.f32(name, n);
        if (!p) return false;
        dst.assign(p, p + n);
        return true;
    };
    if (!cp("table.whisper_mel_128x201", 128 * 201, t.whisper_mel) ||
        !cp("table.kaldi_mel_80x257", 80 * 257, t.kaldi_mel) ||
        !cp("table.whisper_hann400", 400, t.whisper_hann) ||
        !cp("table.kaldi_povey512", 512, t.kaldi_povey) ||
        !cp("table.flow_noise_1x80x15000", 80 * 15000, t.flow_noise) ||
        !cp("table.flow_mel_80x961", 80 * 961, t.flow_mel) ||
        !cp("table.flow_hann1920", 1920, t.flow_hann)) {
        return false;
    }
    // The merged flow kernel keeps its five Euler states in BF16.  The
    // calibration table removes the small, repeatable per-bin gain/bias error
    // without changing the kernel's single-call CFG/Euler contract.
    if (g.get("table.flow_affine")) {
        if (!cp("table.flow_affine", 160, t.flow_affine)) return false;
    }
    return true;
}

static inline float flow_mel_value(const feature_tables & t, int c, float v) {
    if (t.flow_affine.size() != 160) return v;
    return t.flow_affine[(size_t) c] * v + t.flow_affine[(size_t) (80 + c)];
}

// Whisper log-mel [128,T] normalized exactly like whisper.log_mel_spectrogram.
static std::vector<float> whisper_mel(const std::vector<float> & pcm16, int & n_frames, const feature_tables & t) {
    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(400);
    cache.hann_window = t.whisper_hann;
    cache.filters.n_mel = 128;
    cache.filters.n_fft = 400;
    cache.filters.data = t.whisper_mel;

    const int n_fft = 400, hop = 160, pad = n_fft / 2;
    std::vector<float> x(pcm16.size() + 2 * pad);
    for (int i = 0; i < pad; i++) x[i] = pcm16[pad - i];
    std::copy(pcm16.begin(), pcm16.end(), x.begin() + pad);
    for (int i = 0; i < pad; i++) x[pcm16.size() + pad + i] = pcm16[pcm16.size() - 2 - i];

    n_frames = (int) (pcm16.size() / hop);
    std::vector<float> mel((size_t) 128 * n_frames);
    std::vector<float> frame(n_fft), frame_mel(128);
    float vmax = -1e30f;
    for (int i = 0; i < n_frames; i++) {
        std::copy(x.begin() + (size_t) i * hop, x.begin() + (size_t) i * hop + n_fft, frame.begin());
        mtmd_audio_log_mel_frame(cache, frame.data(), n_fft, 128, frame_mel.data());
        for (int j = 0; j < 128; j++) {
            mel[(size_t) j * n_frames + i] = frame_mel[j];
            vmax = std::max(vmax, frame_mel[j]);
        }
    }
    vmax -= 8.0f;
    for (auto & v : mel) v = (std::max(v, vmax) + 4.0f) / 4.0f;
    return mel;
}

// Kaldi fbank [T,80], snip_edges=True, povey window, preemphasis 0.97, 512 FFT.
static std::vector<float> kaldi_fbank80(const std::vector<float> & pcm16, int & n_frames, const feature_tables & t) {
    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(512);
    cache.hann_window = t.kaldi_povey;
    cache.filters.n_mel = 80;
    cache.filters.n_fft = 512;
    cache.filters.data = t.kaldi_mel;

    const int frame_len = 400, shift = 160, n_fft = 512;
    n_frames = pcm16.size() >= frame_len ? 1 + (int) ((pcm16.size() - frame_len) / shift) : 0;
    std::vector<float> feat((size_t) n_frames * 80);
    std::vector<float> frame(n_fft, 0.0f);
    std::vector<double> mean(80, 0.0);
    for (int i = 0; i < n_frames; i++) {
        const float * src = pcm16.data() + (size_t) i * shift;
        double avg = 0;
        for (int j = 0; j < frame_len; j++) { frame[j] = src[j]; avg += frame[j]; }
        avg /= frame_len;
        for (int j = 0; j < frame_len; j++) frame[j] -= (float) avg;
        for (int j = frame_len - 1; j >= 1; j--) frame[j] -= 0.97f * frame[j - 1];
        frame[0] -= 0.97f * frame[0];
        for (int j = frame_len; j < n_fft; j++) frame[j] = 0.0f;
        float * dst = feat.data() + (size_t) i * 80;
        mtmd_audio_log_mel_frame(cache, frame.data(), n_fft, 80, dst);
        for (int j = 0; j < 80; j++) {
            dst[j] = std::max(dst[j] * 2.302585092994046f, std::log(1.1920928955078125e-7f));
            mean[j] += dst[j];
        }
    }
    for (int j = 0; j < 80; j++) mean[j] /= std::max(1, n_frames);
    for (int i = 0; i < n_frames; i++) for (int j = 0; j < 80; j++) feat[(size_t) i * 80 + j] -= (float) mean[j];
    return feat;
}

// CosyVoice3 flow front end: 24 kHz, n_fft 1920, hop 480, reflect pad 720.
static std::vector<float> flow_mel24(const std::vector<float> & pcm24, int & n_frames, const feature_tables & t) {
    mtmd_audio_cache cache;
    cache.fill_sin_cos_table(1920);
    cache.hann_window = t.flow_hann;
    const int n_bins = 1920 / 2 + 1;

    const int n_fft = 1920, hop = 480, pad = (n_fft - hop) / 2;
    std::vector<float> x(pcm24.size() + 2 * pad);
    for (int i = 0; i < pad; i++) x[i] = pcm24[pad - i];
    std::copy(pcm24.begin(), pcm24.end(), x.begin() + pad);
    for (int i = 0; i < pad; i++) x[pcm24.size() + pad + i] = pcm24[pcm24.size() - 2 - i];

    n_frames = (int) (pcm24.size() / hop);
    std::vector<float> feat((size_t) n_frames * 80);
    const float floor_log = std::log(1e-5f);
    parallel_ranges(n_frames, 16, [&](int f0, int f1) {
        std::vector<float> power(n_bins);
        for (int i = f0; i < f1; i++) {
            mtmd_audio_power_frame(cache, x.data() + (size_t) i * hop, n_fft, power.data());
            float * dst = feat.data() + (size_t) i * 80;
            for (int m = 0; m < 80; m++) {
                double sum = 0.0;
                const float * filter = t.flow_mel.data() + (size_t) m * n_bins;
                for (int k = 0; k < n_bins; k++) sum += (double) filter[k] * std::sqrt(power[k] + 1e-9f);
                dst[m] = std::max(std::log((float) sum), floor_log);
            }
        }
    });
    return feat;
}

// ---------------------------------------------------------------------------
// Auxiliary GGUF

struct aux_weights {
    gguf_context * ctx = nullptr;
    ggml_context * tensors = nullptr;
    std::vector<float> speech_emb;   // [6761,896]
    std::vector<float> speech_dec;   // [6761,896]
    int n_speech = 0, n_embd = 0;
};

static bool load_aux(const std::string & path, aux_weights & w) {
    gguf_init_params p{};
    p.no_alloc = false;
    p.ctx = &w.tensors;
    w.ctx = gguf_init_from_file(path.c_str(), p);
    if (!w.ctx) { fprintf(stderr, "failed to load %s\n", path.c_str()); return false; }
    ggml_tensor * se = ggml_get_tensor(w.tensors, "speech_embedding.weight");
    ggml_tensor * sd = ggml_get_tensor(w.tensors, "llm_decoder.weight");
    if (!se || !sd) { fprintf(stderr, "aux GGUF lacks speech_embedding/llm_decoder\n"); return false; }
    if (se->type != GGML_TYPE_F32 || sd->type != GGML_TYPE_F32) { fprintf(stderr, "aux speech tensors must be F32\n"); return false; }
    w.n_speech = (int) se->ne[1];
    w.n_embd = (int) se->ne[0];
    w.speech_emb.resize((size_t) w.n_speech * w.n_embd);
    w.speech_dec.resize((size_t) w.n_speech * w.n_embd);
    memcpy(w.speech_emb.data(), se->data, w.speech_emb.size() * sizeof(float));
    memcpy(w.speech_dec.data(), sd->data, w.speech_dec.size() * sizeof(float));
    return true;
}

// ---------------------------------------------------------------------------
// Sampling: exact CPU torch semantics for CosyVoice3's RAS.  torch's CPU
// generator is MT19937 with a 53-bit double uniform draw; multinomial uses a
// float cumulative distribution and a half-open binary search.  Matching both
// is required for the same seed to produce the official token sequence.

struct torch_rng {
    static constexpr int N = 624, M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu, UMASK = 0x80000000u, LMASK = 0x7fffffffu;
    uint32_t state[N];
    int left = 1, next = 0;

    explicit torch_rng(uint32_t seed) {
        state[0] = seed;
        for (int j = 1; j < N; j++) state[j] = 1812433253u * (state[j - 1] ^ (state[j - 1] >> 30)) + (uint32_t) j;
    }
    static uint32_t twist(uint32_t u, uint32_t v) {
        return (((u & UMASK) | (v & LMASK)) >> 1) ^ ((v & 1u) ? MATRIX_A : 0u);
    }
    void next_state() {
        uint32_t * p = state;
        left = N; next = 0;
        for (int j = N - M + 1; --j; p++) *p = p[M] ^ twist(p[0], p[1]);
        for (int j = M; --j; p++) *p = p[M - N] ^ twist(p[0], p[1]);
        *p = p[M - N] ^ twist(p[0], state[0]);
    }
    uint32_t random() {
        if (--left == 0) next_state();
        uint32_t y = state[next++];
        y ^= y >> 11; y ^= (y << 7) & 0x9d2c5680u; y ^= (y << 15) & 0xefc60000u; y ^= y >> 18;
        return y;
    }
    uint64_t random64() { return ((uint64_t) random() << 32) | random(); }
    double uniform() { return (double) (random64() & ((1ull << 53) - 1)) / (double) (1ull << 53); }
};

struct sampler {
    torch_rng rng;
    bool greedy = false;
    std::vector<float> probs;
    std::vector<float> cum;
    explicit sampler(uint32_t seed, bool greedy_ = false) : rng(seed), greedy(greedy_) {}

    void softmax_into(const std::vector<float> & logits, std::vector<float> & p) {
        const float mx = *std::max_element(logits.begin(), logits.end());
        p.resize(logits.size());
        float sum = 0;
        for (size_t i = 0; i < logits.size(); i++) {
            p[i] = std::isfinite(logits[i]) ? std::exp(logits[i] - mx) : 0.0f;
            sum += p[i];
        }
        if (sum > 0) for (float & v : p) v /= sum;
    }

    int multinomial(const std::vector<float> & p) {
        cum.resize(p.size());
        float sum = 0;
        for (size_t i = 0; i < p.size(); i++) { sum += std::max(0.0f, p[i]); cum[i] = sum; }
        if (!(sum > 0)) return 0;
        for (float & v : cum) v /= sum;
        cum.back() = 1.0f;
        const double u = rng.uniform();
        int left = 0, right = (int) p.size();
        while (right - left > 0) {
            const int mid = left + (right - left) / 2;
            if ((double) cum[mid] < u) left = mid + 1; else right = mid;
        }
        return std::min(left, (int) p.size() - 1);
    }

    int multinomial_prefix(const float * p, int n) {
        std::array<float, 25> local;
        float sum = 0;
        for (int i = 0; i < n; i++) {
            sum += std::max(0.0f, p[i]);
            local[i] = sum;
        }
        if (!(sum > 0)) return 0;
        for (int i = 0; i < n; i++) local[i] /= sum;
        local[n - 1] = 1.0f;
        const double u = rng.uniform();
        int left = 0, right = n;
        while (right - left > 0) {
            const int mid = left + (right - left) / 2;
            if ((double) local[mid] < u) left = mid + 1; else right = mid;
        }
        return std::min(left, n - 1);
    }

    int sample(std::vector<float> logits, const std::vector<int> & decoded, bool ignore_eos, int eos) {
        if (ignore_eos && eos >= 0 && eos < (int) logits.size()) logits[eos] = -std::numeric_limits<float>::infinity();
        softmax_into(logits, probs);
        // Stable descending order is (-probability, index).  Keep only the 25
        // candidates the nucleus can inspect instead of allocating and sorting
        // 6761 indices for every speech token.
        const int ntop = std::min<int>(25, (int) probs.size());
        std::array<int, 25> idx{};
        int top = 0;
        if (greedy) {
            top = (int) (std::max_element(probs.begin(), probs.end()) - probs.begin());
        } else {
            const auto better = [&](int a, int b) {
                return probs[a] > probs[b] || (probs[a] == probs[b] && a < b);
            };
            for (int i = 0; i < ntop; i++) idx[i] = i;
            std::sort(idx.begin(), idx.begin() + ntop, better);
            for (int i = ntop; i < (int) probs.size(); i++) {
                if (!better(i, idx[ntop - 1])) continue;
                idx[ntop - 1] = i;
                for (int j = ntop - 1; j > 0 && better(idx[j], idx[j - 1]); j--) {
                    std::swap(idx[j], idx[j - 1]);
                }
            }
            top = idx[0];
            std::vector<float> nucleus;
            nucleus.reserve(ntop);
            float cum = 0;
            for (int i = 0; i < ntop; i++) {
                if (!(cum < 0.8f) || nucleus.size() >= 25) break;
                cum += probs[idx[i]];
                nucleus.push_back(probs[idx[i]]);
            }
            if (nucleus.empty()) nucleus.push_back(probs[idx[0]]);
            top = idx[multinomial_prefix(nucleus.data(), (int) nucleus.size())];
        }

        int rep = 0;
        for (int i = std::max(0, (int) decoded.size() - 10); i < (int) decoded.size(); i++) rep += decoded[i] == top;
        if (rep >= 1) {
            logits[top] = -std::numeric_limits<float>::infinity();
            if (greedy) {
                top = (int) (std::max_element(logits.begin(), logits.end()) - logits.begin());
            } else {
                softmax_into(logits, probs);
                top = multinomial(probs);
            }
        }
        return top;
    }
};

// ---------------------------------------------------------------------------
// Reference-audio feature cache
//
// The speaker encoder, the speech tokenizer and both mel front ends depend only
// on the prompt wav.  The HTTP server runs one process per request, so the
// cache has to live on disk: a warm request with the same recording skips all
// of them.

struct feat_cache {
    int fbank_frames = 0, wmel_frames = 0, flow_frames = 0, n_tok = 0;
    std::vector<float>   fbank, wmel, flowfeat, embedding;
    std::vector<int32_t> tok;
};

struct ref_feats {
    int fbank_frames = 0, wmel_frames = 0, flow_frames = 0;
    std::vector<float> fbank, wmel, flowfeat, embedding;
    std::vector<int32_t> tok;
};

static uint64_t feat_hash(const void * data, size_t bytes) {
    const uint8_t * p = (const uint8_t *) data;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i < bytes; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static std::string feat_cache_path(const std::string & dir, uint64_t h) {
    char name[64];
    snprintf(name, sizeof(name), "feat_%016llx.bin", (unsigned long long) h);
    return dir + "/" + name;
}

static bool feat_cache_load(const std::string & path, feat_cache & c) {
    FILE * f = fopen(path.c_str(), "rb");
    if (!f) return false;
    char magic[8];
    uint32_t hdr[4];
    bool ok = fread(magic, 1, 8, f) == 8 && memcmp(magic, "COSYFT2", 7) == 0 &&
              fread(hdr, sizeof(uint32_t), 4, f) == 4;
    if (ok) {
        c.fbank_frames = (int) hdr[0];
        c.wmel_frames  = (int) hdr[1];
        c.flow_frames  = (int) hdr[2];
        c.n_tok        = (int) hdr[3];
        auto rd = [&](std::vector<float> & v, size_t n) {
            v.resize(n);
            return n == 0 || fread(v.data(), sizeof(float), n, f) == n;
        };
        ok = rd(c.fbank, (size_t) c.fbank_frames * 80) && rd(c.wmel, (size_t) 128 * c.wmel_frames) &&
             rd(c.flowfeat, (size_t) c.flow_frames * 80) && rd(c.embedding, 192);
        if (ok) {
            c.tok.resize((size_t) c.n_tok);
            ok = c.n_tok > 0 && fread(c.tok.data(), sizeof(int32_t), (size_t) c.n_tok, f) == (size_t) c.n_tok;
        }
    }
    fclose(f);
    return ok;
}

static void feat_cache_save(const std::string & path, const feat_cache & c) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return;
    uint32_t hdr[4] = { (uint32_t) c.fbank_frames, (uint32_t) c.wmel_frames, (uint32_t) c.flow_frames, (uint32_t) c.n_tok };
    fwrite("COSYFT2", 1, 8, f);
    fwrite(hdr, sizeof(uint32_t), 4, f);
    fwrite(c.fbank.data(), sizeof(float), c.fbank.size(), f);
    fwrite(c.wmel.data(), sizeof(float), c.wmel.size(), f);
    fwrite(c.flowfeat.data(), sizeof(float), c.flowfeat.size(), f);
    fwrite(c.embedding.data(), sizeof(float), c.embedding.size(), f);
    fwrite(c.tok.data(), sizeof(int32_t), c.tok.size(), f);
    fclose(f);
}

// ---------------------------------------------------------------------------
// Main

struct args {
    std::string model, aux, front;
    std::string flow_npu, flow_npu_long, flow_loop_npu, hift_npu, front_npu;
    bool flow_npu_nlc = false, flow_npu_long_nlc = false;
    int flow_npu_cap = 512, flow_npu_long_cap = 1024;
    int flow_prompt_min = 32;
    int flow_steps = 10;
    int hift_left = 64;
    std::string text, prompt_text, prompt_wav, out = "out.wav";
    int ngl = 0, ctx = 4096, threads = 16;
    uint32_t seed = 1986;
    bool greedy = false;
    bool dump_codes = false, mel_only = false, feat_only = false;
    bool worker = false;
    int repeat = 1;
    std::string parity_in, parity_out, dump_feats, codes_in, dump_mel, dump_hift_s, dump_hift_mp;
    std::string feat_cache;
    std::string warm_prompt, warm_text, warm_prompt2, warm_text2;
};

static bool parse_args(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto val = [&]() -> std::string { if (++i >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(1); } return argv[i]; };
        if (k == "-m" || k == "--model") a.model = val();
        else if (k == "--aux") a.aux = val();
        else if (k == "--front") a.front = val();
        else if (k == "--front-npu") a.front_npu = val();
        else if (k == "--flow-npu") a.flow_npu = val();
        else if (k == "--flow-loop-npu") a.flow_loop_npu = val();
        else if (k == "--flow-prompt-min") a.flow_prompt_min = std::max(0, std::stoi(val()));
        else if (k == "--flow-steps") a.flow_steps = std::max(1, std::min(32, std::stoi(val())));
        else if (k == "--hift-left") a.hift_left = std::max(8, std::stoi(val()));
        else if (k == "--flow-npu-long") a.flow_npu_long = val();
        else if (k == "--hift-npu") a.hift_npu = val();
        else if (k == "--flow-npu-nlc") a.flow_npu_nlc = true;
        else if (k == "--flow-npu-long-nlc") a.flow_npu_long_nlc = true;
        else if (k == "--flow-npu-cap") a.flow_npu_cap = std::stoi(val());
        else if (k == "--flow-npu-long-cap") a.flow_npu_long_cap = std::stoi(val());
        else if (k == "-p" || k == "--text") a.text = val();
        else if (k == "--prompt-text") a.prompt_text = val();
        else if (k == "--prompt-wav") a.prompt_wav = val();
        else if (k == "-o" || k == "--out") a.out = val();
        else if (k == "-ngl") a.ngl = std::stoi(val());
        else if (k == "-c") a.ctx = std::stoi(val());
        else if (k == "-t") a.threads = std::stoi(val());
        else if (k == "--seed") a.seed = (uint32_t) std::stoul(val());
        else if (k == "-g" || k == "--greedy") a.greedy = true;
        else if (k == "--dump-codes") a.dump_codes = true;
        else if (k == "--mel-only") a.mel_only = true;
        else if (k == "--feat-only") a.feat_only = true;
        else if (k == "--worker") a.worker = true;
        else if (k == "--repeat") a.repeat = std::max(1, std::stoi(val()));
        else if (k == "--parity-in") a.parity_in = val();
        else if (k == "--parity-out") a.parity_out = val();
        else if (k == "--dump-feats") a.dump_feats = val();
        else if (k == "--codes") a.codes_in = val();
        else if (k == "--dump-mel") a.dump_mel = val();
        else if (k == "--dump-hift-s") a.dump_hift_s = val();
        else if (k == "--dump-hift-mp") a.dump_hift_mp = val();
        else if (k == "--feat-cache") a.feat_cache = val();
        else if (k == "--warm-prompt-wav") a.warm_prompt = val();
        else if (k == "--warm-prompt-text") a.warm_text = val();
        else if (k == "--warm-prompt-wav2") a.warm_prompt2 = val();
        else if (k == "--warm-prompt-text2") a.warm_text2 = val();
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return false; }
    }
    if (a.feat_only) return !a.front.empty() && !a.prompt_wav.empty();
    return !a.model.empty() && !a.aux.empty() && !a.front.empty() && !a.front_npu.empty() &&
           !a.text.empty() && !a.prompt_wav.empty() && !a.flow_loop_npu.empty() && !a.hift_npu.empty();
}

static bool g_ax_engine_ready = false;

static bool axera_engine_init_once() {
#ifdef COSY_TTS_HAVE_AXERA
    if (g_ax_engine_ready) return true;
    static const int rc = [] {
        if (AX_SYS_Init() != 0) return 1;
        AX_ENGINE_NPU_ATTR_T attr{};
        attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        return AX_ENGINE_Init(&attr) == 0 ? 0 : 2;
    }();
    if (rc) fprintf(stderr, "AXERA engine init failed (%d)\n", rc);
    g_ax_engine_ready = rc == 0;
    return g_ax_engine_ready;
#else
    fprintf(stderr, "this build has no AXERA support; rebuild with GGML_AXERA and GGML_AXERA_BSP_DIR\n");
    return false;
#endif
}

static double sec(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static thread_local bool tl_range_worker = false;

// One persistent team for the whole process.  The reference front end issues
// hundreds of small convolutions; creating a sixteen-thread team for each one
// costs more than the arithmetic on A55.
class range_pool {
public:
    range_pool(int hw, int workers) : stop_(false), generation_(0), n_(0), nt_(1), finished_(0) {
        (void) hw;
        for (int i = 1; i < workers; i++) threads_.emplace_back([this, i] { worker(i); });
        std::unique_lock<std::mutex> lk(state_mu_);
        ready_cv_.wait(lk, [&] { return started_ == workers - 1; });
    }
    void run(int n, int nt, std::function<void(int, int)> f) {
        std::unique_lock<std::mutex> job_lock(job_mu_);
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            job_ = std::move(f);
            n_ = n; nt_ = nt; finished_ = 0;
            generation_++;
        }
        work_cv_.notify_all();
        if (getenv("COSY_PROFILE")) fprintf(stderr, "[pool] run n=%d nt=%d\n", n, nt);
        job_(0, n / nt);
        {
            std::unique_lock<std::mutex> lk(state_mu_);
            done_cv_.wait(lk, [&] { return finished_ >= nt - 1; });
        }
        if (getenv("COSY_PROFILE")) fprintf(stderr, "[pool] done nt=%d\n", nt);
    }
private:
    void worker(int idx) {
        tl_range_worker = true;
        uint64_t seen = 0;
        {
            std::lock_guard<std::mutex> lk(state_mu_);
            started_++;
        }
        ready_cv_.notify_one();
        for (;;) {
            int n, nt;
            {
                std::unique_lock<std::mutex> lk(state_mu_);
                work_cv_.wait(lk, [&] { return stop_ || generation_ != seen; });
                if (stop_) return;
                seen = generation_;
                n = n_; nt = nt_;
            }
            if (idx < nt) {
                if (getenv("COSY_PROFILE")) fprintf(stderr, "[pool] worker %d range %d..%d n=%d nt=%d\n", idx, (n*idx)/nt, (n*(idx+1))/nt, n, nt);
                job_((n * idx) / nt, (n * (idx + 1)) / nt);
                {
                    std::lock_guard<std::mutex> lk(state_mu_);
                    finished_++;
                }
                done_cv_.notify_one();
            }
        }
    }
    std::mutex job_mu_, state_mu_;
    std::condition_variable work_cv_, done_cv_, ready_cv_;
    std::function<void(int, int)> job_;
    int n_, nt_, finished_, started_ = 0;
    uint64_t generation_;
    bool stop_;
    std::vector<std::thread> threads_;
};

template <typename Fn>
static void parallel_ranges(int n, int max_threads, Fn && fn) {
    if (n <= 0) return;
    static const int hw_threads = std::max(1u, std::thread::hardware_concurrency());
    const int nt = std::max(1, std::min(std::min(max_threads, n), hw_threads));
    if (nt == 1 || tl_range_worker) { fn(0, n); return; }
    static range_pool * pool = new range_pool(hw_threads, hw_threads);
    pool->run(n, nt, std::function<void(int, int)>(std::forward<Fn>(fn)));
}

struct cosy_service_request {
    std::string text;
    std::string ref_text;
    std::vector<uint8_t> ref_wav;
    uint32_t seed = 1986;
    bool streaming = false;
    cosy_audio_chunk_callback on_chunk;
    std::vector<uint8_t> result;
    std::string error;
    bool ok = false;
    bool done = false;
};

struct cosy_service_control {
    std::mutex mu;
    std::condition_variable cv;
    std::thread thread;
    std::unique_ptr<cosy_service_request> request;
    bool ready = false;
    bool starting = false;
    bool failed = false;
    bool stopping = false;
    std::string start_error;
};

static cosy_service_control g_service;

struct llama_batch_guard {
    llama_batch batch{};
    llama_batch_guard(int capacity, int n_embd) : batch(llama_batch_init(capacity, n_embd, 1)) {}
    llama_batch_guard(const llama_batch_guard &) = delete;
    llama_batch_guard & operator=(const llama_batch_guard &) = delete;
    ~llama_batch_guard() { llama_batch_free(batch); }
};

static int decode_embd(llama_context * ctx, llama_batch & batch, const float * embd,
                       int n_tokens, int n_embd, int pos0, bool logits_last) {
    batch.n_tokens = n_tokens;
    memcpy(batch.embd, embd, (size_t) n_tokens * n_embd * sizeof(float));
    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits_last && i == n_tokens - 1;
    }
    return llama_decode(ctx, batch);
}

static int cosy_tts_main_impl(int argc, char ** argv, bool service_mode) {
    // llama-server's AXERA backend has already performed the process-global
    // AX_SYS/AX_ENGINE initialization.  Calling Init a second time from this
    // runtime corrupts the shared engine state.  The standalone CLI still owns
    // the initialization because it has no other AXERA backend user.
    g_ax_engine_ready = service_mode;
    args a;
    if (!parse_args(argc, argv, a)) {
        fprintf(stderr,
            "usage: %s -m llm.gguf --aux aux.gguf --front front.gguf --front-npu front.axmodel\n"
            "  --flow-loop-npu flow.axmodel --hift-npu hift.axmodel\n"
            "  -p TEXT --prompt-text TEXT --prompt-wav ref.wav -o out.wav [-ngl N] [--dump-codes]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    auto t_all0 = std::chrono::steady_clock::now();
    aux_weights aux;
    auto t_aux0 = std::chrono::steady_clock::now();
    if (!load_aux(a.aux, aux)) return 1;
    auto t_aux1 = std::chrono::steady_clock::now();

    // LLM-only parity check: no front end, ONNX session or flow/HiFT model is
    // constructed.  The file is [rows, n_embd] f32 embeddings; every row's
    // final hidden state is written back for comparison with the CPU path.
    if (!a.parity_in.empty()) {
        llama_model_params mp = llama_model_default_params();
        mp.n_gpu_layers = a.ngl;
        llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
        if (!model) { fprintf(stderr, "failed to load %s\n", a.model.c_str()); return 1; }
        llama_context_params cp = llama_context_default_params();
        cp.n_ctx = a.ctx;
        cp.n_batch = 512;
        cp.n_ubatch = 256;
        cp.n_threads = a.threads;
        cp.n_threads_batch = a.threads;
        cp.embeddings = true;
        cp.offload_kqv = true;
        llama_context * ctx = llama_init_from_model(model, cp);
        if (!ctx) { fprintf(stderr, "failed to create llama context\n"); return 1; }
        const int n_embd = llama_model_n_embd(model);
        if (n_embd != aux.n_embd) { fprintf(stderr, "embedding dim mismatch %d vs %d\n", n_embd, aux.n_embd); return 1; }
        auto in = load_f32(a.parity_in);
        const int rows = (int) (in.size() / n_embd);
        if (rows <= 0 || (size_t) rows * n_embd != in.size()) { fprintf(stderr, "bad parity input\n"); return 1; }
        llama_batch_guard batch(rows, n_embd);
        if (decode_embd(ctx, batch.batch, in.data(), rows, n_embd, 0, true) != 0) { fprintf(stderr, "parity decode failed\n"); return 1; }
        std::vector<float> all_hidden((size_t) rows * n_embd);
        for (int r = 0; r < rows; r++) {
            const float * h = llama_get_embeddings_ith(ctx, r);
            if (!h) { fprintf(stderr, "no hidden row %d\n", r); return 1; }
            memcpy(all_hidden.data() + (size_t) r * n_embd, h, (size_t) n_embd * sizeof(float));
        }
        const float * hidden = llama_get_embeddings_ith(ctx, -1);
        std::vector<float> logits(aux.n_speech);
        for (int j = 0; j < aux.n_speech; j++) {
            const float * w = aux.speech_dec.data() + (size_t) j * n_embd;
            double sum = 0;
            for (int k = 0; k < n_embd; k++) sum += (double) w[k] * hidden[k];
            logits[j] = (float) sum;
        }
        auto dump = [&](const std::string & path, const float * data, size_t n) {
            FILE * f = fopen(path.c_str(), "wb"); fwrite(data, 4, n, f); fclose(f);
        };
        dump(a.parity_out + ".hidden.f32", hidden, n_embd);
        dump(a.parity_out + ".all_hidden.f32", all_hidden.data(), all_hidden.size());
        dump(a.parity_out + ".logits.f32", logits.data(), logits.size());
        fprintf(stderr, "parity rows=%d hidden=%d logits=%d\n", rows, n_embd, aux.n_speech);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }

    gguf_asset_file front_assets;
    if (!front_assets.open(a.front)) return 1;
    feature_tables tab;
    if (!load_tables_gguf(front_assets, tab)) return 1;
    hift_native native_hift;
    if (!native_hift.load(front_assets)) return 1;
    campplus_net campplus;
    if (!campplus.load(front_assets)) return 1;
    if (a.feat_only) {
        // Development aid: time only the host reference-audio front ends.
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto el = [&](auto a0, auto b0) { return sec(a0, b0); };
        wav_data wav;
        if (!read_wav(a.prompt_wav.c_str(), wav)) return 1;
        auto w0 = now();
        auto pcm16 = resample_sinc(wav.pcm, wav.sample_rate, 16000);
        auto pcm24 = resample_sinc(wav.pcm, wav.sample_rate, 24000);
        auto w1 = now();
        int nf = 0, nw = 0, nm = 0;
        auto fb = kaldi_fbank80(pcm16, nf, tab);
        auto w2 = now();
        std::vector<double> ct;
        auto emb = campplus.forward(fb, nf, "ref", &ct);
        auto w3 = now();
        auto wm = whisper_mel(pcm16, nw, tab);
        auto w4 = now();
        auto fm = flow_mel24(pcm24, nm, tab);
        auto w5 = now();
        fprintf(stderr, "[feat] resample=%.3f fbank=%.3f speaker=%.3f wmel=%.3f flow_mel=%.3f total=%.3f "
                "fbank_T=%d wmel_T=%d flow_T=%d\n",
                el(w0,w1), el(w1,w2), el(w2,w3), el(w3,w4), el(w4,w5), el(w0,w5), nf, nw, nm);
        if (ct.size() == 3) fprintf(stderr, "[campplus] head=%.3f tdnn=%.3f stats=%.3f\n", ct[0], ct[1], ct[2]);
        if (!a.dump_feats.empty()) {
            auto dump = [&](const char * suffix, const void * data, size_t bytes) {
                FILE * f = fopen((a.dump_feats + suffix).c_str(), "wb"); fwrite(data, 1, bytes, f); fclose(f);
            };
            dump(".fbank.f32", fb.data(), fb.size() * 4);
            dump(".wmel.f32", wm.data(), wm.size() * 4);
            dump(".flowfeat.f32", fm.data(), fm.size() * 4);
            dump(".embedding.f32", emb.data(), emb.size() * 4);
        }
        return 0;
    }
    const bool have_native_hift = true;
    double t_sess = 0.0;
    std::unique_ptr<npu_model> front_npu;
    std::unique_ptr<npu_model> flow_npu, flow_npu_long, flow_loop_npu, hift_npu;
    double t_npu_load = 0.0;
    if (!a.flow_loop_npu.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        flow_loop_npu = std::make_unique<npu_model>();
        if (!flow_loop_npu->load(a.flow_loop_npu)) return 1;
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    }
    if (!a.front_npu.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        front_npu = std::make_unique<npu_model>();
        if (!front_npu->load(a.front_npu)) return 1;
        const double load_s = sec(s0, std::chrono::steady_clock::now());
        auto s1 = std::chrono::steady_clock::now();
        if (!upload_front_weights(*front_npu, front_assets)) return 1;
        fprintf(stderr, "[front-npu] session=%.3fs upload=%.3fs\n", load_s,
                sec(s1, std::chrono::steady_clock::now()));
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    }
    // The merged flow graph may expose the Euler state under its own name
    // (the debug build materializes x1..x5 to force inter-step ordering).
    // Prefer the named final state over output index 0, which can be mu.
    int flow_out_idx = 0;
    if (flow_loop_npu) {
        for (const char * name : {"mel", "x5", "output"}) {
            const int idx = flow_loop_npu->output_index(name);
            if (idx >= 0) { flow_out_idx = idx; break; }
        }
    }
    if (!a.flow_npu.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        flow_npu = std::make_unique<npu_model>();
        if (!flow_npu->load(a.flow_npu)) return 1;
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    }
    if (!a.flow_npu_long.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        flow_npu_long = std::make_unique<npu_model>();
        if (!flow_npu_long->load(a.flow_npu_long)) return 1;
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    }
    if (!a.hift_npu.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        hift_npu = std::make_unique<npu_model>();
        if (!hift_npu->load(a.hift_npu)) return 1;
        // The first launch of the 89 MiB convolution bank can return stale
        // output rows.  A synthetic launch at load time moves that one-time
        // device transition out of the first speech request.
        const int warm_cap = (int) (hift_npu->input_elems("mel") / 80);
        std::vector<float> wm((size_t) 80 * warm_cap), ws((size_t) 18 * (warm_cap * 120 + 1));
        for (size_t i = 0; i < wm.size(); i++) wm[i] = 0.25f * std::sin((float) i * 0.013f);
        for (size_t i = 0; i < ws.size(); i++) ws[i] = 0.1f * std::cos((float) i * 0.007f);
        if (!hift_npu->set_input("mel", wm.data(), wm.size() * 4) ||
            !hift_npu->set_input("s_stft", ws.data(), ws.size() * 4) || !hift_npu->run()) return 1;
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    }

    // ---- reference audio front ends -------------------------------------
    std::string cache_dir = a.feat_cache;
    if (cache_dir.empty()) {
        const char * tmp = getenv("TMPDIR");
        cache_dir = std::string(tmp && *tmp ? tmp : "/tmp") + "/cosy_feat_cache";
    }
    mkdir(cache_dir.c_str(), 0777);
    std::unordered_map<uint64_t, std::shared_ptr<ref_feats>> ref_mem;
    auto prepare_ref = [&](const std::string & wavpath) -> std::shared_ptr<ref_feats> {
        auto now = [] { return std::chrono::steady_clock::now(); };
        auto ts0 = now();
        wav_data wav;
        if (!read_wav(wavpath.c_str(), wav)) return nullptr;
        // Hit the in-process cache before resampling.  The HTTP worker writes
        // each request's reference bytes to a fresh temporary path, so keying
        // on the decoded PCM (not the path) is what makes repeated prompts
        // genuinely free even when their file names differ.
        const uint64_t mem_key = feat_hash(wav.pcm.data(), wav.pcm.size() * sizeof(float)) ^
                                 ((uint64_t) wav.pcm.size() * 0x9e3779b97f4a7c15ull) ^
                                 ((uint64_t) wav.sample_rate << 32);
        auto hit = ref_mem.find(mem_key);
        if (hit != ref_mem.end()) {
            fprintf(stderr, "[frontend] cache=memory total=%.3fs speech_tokens=%zu\n",
                    sec(ts0, now()), hit->second->tok.size());
            return hit->second;
        }
        auto pcm16 = resample_sinc(wav.pcm, wav.sample_rate, 16000);
        auto pcm24 = resample_sinc(wav.pcm, wav.sample_rate, 24000);
        const uint64_t feat_key = feat_hash(pcm16.data(), pcm16.size() * sizeof(float)) ^
                                  ((uint64_t) pcm16.size() * 0x9e3779b97f4a7c15ull);
        auto rf = std::make_shared<ref_feats>();
        const std::string feat_path = feat_cache_path(cache_dir, feat_key);
        feat_cache fc;
        const bool feat_hit = feat_cache_load(feat_path, fc);
        double t_fbank = 0, t_spk = 0, t_wmel = 0, t_stok = 0, t_flowmel = 0;
        if (feat_hit) {
            rf->fbank_frames = fc.fbank_frames;
            rf->wmel_frames  = fc.wmel_frames;
            rf->flow_frames  = fc.flow_frames;
            rf->fbank        = std::move(fc.fbank);
            rf->wmel         = std::move(fc.wmel);
            rf->flowfeat     = std::move(fc.flowfeat);
            rf->embedding    = std::move(fc.embedding);
            rf->tok          = std::move(fc.tok);
        } else {
            auto t0 = now();
            // The two spectrogram front ends are independent; overlap them.
            std::thread wmel_thread([&] { rf->wmel = whisper_mel(pcm16, rf->wmel_frames, tab); });
            rf->fbank = kaldi_fbank80(pcm16, rf->fbank_frames, tab);
            wmel_thread.join();
            auto t1 = now();
            if (!front_npu) {
                fprintf(stderr, "reference-audio tokenizer AXModel was not loaded\n");
                return nullptr;
            }
            // The tokenizer runs on the NPU while Campplus uses the host CPU.
            // Both are independent and the session is not touched again until
            // the thread is joined.
            bool tok_ok = false;
            std::thread tok_thread([&] {
                tok_ok = front_tokenize(*front_npu, rf->wmel, rf->wmel_frames, rf->tok);
            });
            std::vector<double> cam_t;
            rf->embedding = campplus.forward(rf->fbank, rf->fbank_frames, "ref", &cam_t);
            if (getenv("COSY_PROFILE") && cam_t.size() == 3) {
                fprintf(stderr, "[campplus] head=%.3fs tdnn=%.3fs stats=%.3fs\n", cam_t[0], cam_t[1], cam_t[2]);
            }
            tok_thread.join();
            auto t2 = now();
            if (!tok_ok) return nullptr;
            if ((int) rf->embedding.size() != 192) {
                fprintf(stderr, "Campplus produced %zu values\n", rf->embedding.size());
                return nullptr;
            }
            rf->flowfeat = flow_mel24(pcm24, rf->flow_frames, tab);
            auto t5 = now();
            t_fbank   = sec(t0, t1);
            t_spk     = sec(t0, t2);
            t_wmel    = 0.0;
            t_stok    = sec(t0, t2);
            t_flowmel = sec(t2, t5);
            fc.fbank_frames = rf->fbank_frames; fc.wmel_frames = rf->wmel_frames; fc.flow_frames = rf->flow_frames;
            fc.n_tok = (int) rf->tok.size();
            fc.fbank = rf->fbank; fc.wmel = rf->wmel; fc.flowfeat = rf->flowfeat;
            fc.embedding = rf->embedding; fc.tok = rf->tok;
            feat_cache_save(feat_path, fc);
        }
        auto t_front_done = now();
        fprintf(stderr,
                "[frontend] cache=%s aux_load=%.3fs sessions=%.3fs npu_load=%.3fs total=%.3fs | "
                "fbank=%.3fs speaker=%.3fs whisper_mel=%.3fs tokenizer=%.3fs flow_mel=%.3fs | "
                "fbank_frames=%d whisper_mel_frames=%d flow_mel_frames=%d speech_tokens=%zu\n",
                feat_hit ? "file" : "miss", sec(t_aux0, t_aux1), t_sess, t_npu_load,
                sec(ts0, t_front_done), t_fbank, t_spk, t_wmel, t_stok, t_flowmel,
                rf->fbank_frames, rf->wmel_frames, rf->flow_frames, rf->tok.size());
        if (!a.dump_feats.empty()) {
            auto dump = [&](const std::string & suffix, const void * data, size_t bytes) {
                FILE * f = fopen((a.dump_feats + suffix).c_str(), "wb"); fwrite(data, 1, bytes, f); fclose(f);
            };
            dump(".fbank.f32", rf->fbank.data(), rf->fbank.size() * 4);
            dump(".wmel.f32", rf->wmel.data(), rf->wmel.size() * 4);
            dump(".ptok.i32", rf->tok.data(), rf->tok.size() * 4);
            dump(".embedding.f32", rf->embedding.data(), rf->embedding.size() * 4);
            dump(".flowfeat.f32", rf->flowfeat.data(), rf->flowfeat.size() * 4);
            FILE * f = fopen((a.dump_feats + ".shape.txt").c_str(), "w");
            fprintf(f, "fbank %d 80\nwmel 128 %d\nptok %zu\nflowfeat %d 80\n",
                    rf->fbank_frames, rf->wmel_frames, rf->tok.size(), rf->flow_frames);
            fclose(f);
        }
        ref_mem.emplace(mem_key, rf);
        return rf;
    };

    // ---- Qwen2 backbone -------------------------------------------------
    // The reference prompt (sos + prompt text + task id + prompt speech
    // tokens) is identical for repeated requests with the same reference and
    // text.  Keep its KV in sequence 1 and copy it back instead of re-prefilling
    // ~100 rows every request.  The AXERA backend stores the page images in
    // core/seqstate, so seq_cp moves the private pages too.  This cache must
    // live outside the per-request lambda or it can never hit.
    struct prompt_kv_cache {
        std::vector<int32_t> ids;
        int rows = 0;
        std::vector<float> logits;
        std::vector<uint8_t> state;
        uint64_t used = 0;
    };
    // A request's text precedes the reference-speech prompt, so the cached KV
    // is keyed by both.  Keep a few complete sequence images instead of only
    // the last one: callers commonly alternate voices or repeat an earlier
    // sentence, and a one-entry cache turns both into a full prompt prefill.
    std::vector<prompt_kv_cache> pcache;
    uint64_t pcache_tick = 0;
    constexpr size_t PCACHE_MAX = 8;
    auto t_load0 = std::chrono::steady_clock::now();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = a.ngl;
    llama_model * model = llama_model_load_from_file(a.model.c_str(), mp);
    if (!model) { fprintf(stderr, "failed to load %s\n", a.model.c_str()); return 1; }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = a.ctx;
    cp.n_batch = 512;
    cp.n_ubatch = 256;
    cp.n_threads = a.threads;
    cp.n_threads_batch = a.threads;
    cp.embeddings = true;
    cp.offload_kqv = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create llama context\n"); return 1; }
    auto t_load1 = std::chrono::steady_clock::now();
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    if (n_embd != aux.n_embd) { fprintf(stderr, "embedding dim mismatch %d vs %d\n", n_embd, aux.n_embd); return 1; }

    // ggml_backend_tensor_get on a mapped GGUF has per-call overhead; looking
    // up every prompt row separately made a short prefill look like model
    // compute.  Pull the text table once and index it from host memory.
    std::vector<float> text_emb_table;
    const uint32_t text_emb_elems = llama_model_get_tok_embd(model, nullptr);
    if (text_emb_elems > 0) {
        const auto t0 = std::chrono::steady_clock::now();
        text_emb_table.resize(text_emb_elems);
        if (llama_model_get_tok_embd(model, text_emb_table.data()) != text_emb_elems) {
            fprintf(stderr, "failed to preload the text embedding table\n");
            return 1;
        }
        fprintf(stderr, "[llm] embedding table %.3fs %.1f MiB rows=%u\n",
                sec(t0, std::chrono::steady_clock::now()), text_emb_table.size() * 4.0 / (1024.0 * 1024.0),
                text_emb_elems / (uint32_t) n_embd);
    }

    auto tokenize = [&](const std::string & text) {
        std::vector<llama_token> v(text.size() + 64);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), v.data(), (int) v.size(), false, true);
        if (n < 0) { v.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int) text.size(), v.data(), (int) v.size(), false, true); }
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); std::exit(1); }
        v.resize(n);
        return v;
    };
    auto generate = [&](const std::string & req_text, const std::string & req_prompt_text,
                        const std::string & outpath, const ref_feats & rf, uint32_t req_seed,
                        const std::string & codes_path, double front_s, double llm_load_s,
                        bool incremental, const cosy_audio_chunk_callback & stream_cb) -> int {
    auto t_req0 = std::chrono::steady_clock::now();
    llama_memory_t kv_mem = llama_get_memory(ctx);
    const auto t_kv_rm0 = std::chrono::steady_clock::now();
    llama_memory_seq_rm(kv_mem, 0, -1, -1);
    const double t_kv_rm = sec(t_kv_rm0, std::chrono::steady_clock::now());
    const std::vector<int32_t> & prompt_tok = rf.tok;
    const std::vector<float> & prompt_feat = rf.flowfeat;
    const std::vector<float> & embedding = rf.embedding;
    const auto t_tok0 = std::chrono::steady_clock::now();
    auto prompt_text_tok = tokenize(req_prompt_text);
    auto text_tok = tokenize(req_text);
    const double t_tokenize = sec(t_tok0, std::chrono::steady_clock::now());
    std::vector<llama_token> text_all = prompt_text_tok;
    text_all.insert(text_all.end(), text_tok.begin(), text_tok.end());

    const auto t_embed0 = std::chrono::steady_clock::now();
    std::vector<float> text_emb((size_t) text_all.size() * n_embd);
    const int text_emb_rows = (int) (text_emb_table.size() / (size_t) n_embd);
    if (text_emb_rows == 0) {
        fprintf(stderr, "the model has no preloaded text embedding table\n");
        return 1;
    }
    for (size_t i = 0; i < text_all.size(); i++) {
        if (text_all[i] < 0 || text_all[i] >= text_emb_rows) {
            fprintf(stderr, "text token %d is outside the embedding table\n", text_all[i]);
            return 1;
        }
        memcpy(text_emb.data() + i * n_embd, text_emb_table.data() + (size_t) text_all[i] * n_embd,
               (size_t) n_embd * sizeof(float));
    }

    auto speech_row = [&](int id) { return aux.speech_emb.data() + (size_t) id * n_embd; };
    std::vector<float> lm_input;
    lm_input.insert(lm_input.end(), speech_row(6561), speech_row(6561) + n_embd); // sos
    lm_input.insert(lm_input.end(), text_emb.begin(), text_emb.end());
    lm_input.insert(lm_input.end(), speech_row(6563), speech_row(6563) + n_embd); // task id
    for (int32_t id : prompt_tok) lm_input.insert(lm_input.end(), speech_row(id), speech_row(id + 1));
    const int n_prompt_rows = (int) (lm_input.size() / n_embd);
    const double t_embed = sec(t_embed0, std::chrono::steady_clock::now());
    fprintf(stderr, "[llm] prompt_text=%zu text=%zu prompt_speech=%zu rows=%d\n",
            prompt_text_tok.size(), text_tok.size(), prompt_tok.size(), n_prompt_rows);

    std::vector<int32_t> prefix_key;
    prefix_key.reserve(text_all.size() + prompt_tok.size());
    for (auto t : text_all) prefix_key.push_back((int32_t) t);
    for (auto t : prompt_tok) prefix_key.push_back(t);
    bool prefix_hit = false;
    size_t prefix_hit_idx = 0;
    const auto t_kv_hit0 = std::chrono::steady_clock::now();
    if (codes_path.empty()) {
        for (size_t i = 0; i < pcache.size(); ++i) {
            if (pcache[i].rows == n_prompt_rows && pcache[i].ids == prefix_key) {
                const size_t got = llama_state_seq_set_data(
                        ctx, pcache[i].state.data(), pcache[i].state.size(), 0);
                if (got == pcache[i].state.size()) {
                    prefix_hit = true;
                    prefix_hit_idx = i;
                    pcache[i].used = ++pcache_tick;
                }
                break;
            }
        }
    }
    if (codes_path.empty() && getenv("COSY_LLM_PROBE")) {
        fprintf(stderr, "[llm-cache] entries=%zu rows=%d hit=%d key=%zu\n",
                pcache.size(), n_prompt_rows, prefix_hit ? 1 : 0, prefix_key.size());
    }
    const double t_kv_hit = sec(t_kv_hit0, std::chrono::steady_clock::now());

    std::vector<float> h;
    llama_batch_guard batch(512, n_embd);
    const bool streaming = (bool) stream_cb && incremental;
    const int est_tok_cap = flow_npu ? (int) (flow_npu->input_elems("x") / (2 * 80)) : 0;
    const int stream_tok_cap = flow_loop_npu ? (int) flow_loop_npu->input_elems("tokens") : est_tok_cap;
    const int stream_prompt_n = std::min((int) prompt_tok.size(), std::max(0, stream_tok_cap - 48));
    const int stream_overlap = 4;
    const int stream_chunk_cap = std::max(1, stream_tok_cap - stream_prompt_n - stream_overlap);
    std::vector<int> stream_flow_codes;
    int stream_out_codes = 0;
    int stream_silent_run = 0;
    std::vector<std::vector<float>> stream_mel_ch(80);
    std::vector<double> stream_phase_frames(9, 0.0);
    double stream_flow_s = 0.0, stream_front_s = 0.0, stream_hift_s = 0.0, stream_post_s = 0.0;
    double stream_first_s = -1.0;
    int stream_hift_calls = 0;
    std::string stream_error;
    auto run_stream_window = [&]() -> bool {
        const int available = (int) stream_flow_codes.size();
        if (stream_out_codes >= available) return true;
        if (!flow_loop_npu || !hift_npu || !have_native_hift) {
            stream_error = "streaming requires the merged flow kernel and native HiFT";
            return false;
        }
        // CosyVoice3's streaming path re-runs the complete prefix for every
        // token chunk and returns only the newly produced mel frames.  Keeping
        // the prompt region frozen at its initial noise makes the prompt
        // condition wrong, so the whole fixed-capacity window follows the
        // Euler trajectory.  The merged kernel owns CFG and all Euler updates,
        // so one call produces one complete prefix window.  Long utterances
        // slide that window and keep the prompt at its head.
        const int tok_cap = (int) flow_loop_npu->input_elems("tokens");
        const int mel_cap = (int) (flow_loop_npu->input_elems("noise") / 80);
        if (tok_cap <= stream_prompt_n + 16 || mel_cap != 2 * tok_cap) {
            stream_error = "streaming flow window is too small";
            return false;
        }
        const int max_codes = tok_cap - stream_prompt_n - 4;
        const int begin = std::max(0, available - max_codes);
        const int G = available - begin;
        const int Tw = 2 * (stream_prompt_n + G);
        const int new_start = 2 * (stream_out_codes - begin);
        const int new_frames = 2 * (available - stream_out_codes);
        const int total_tokens = stream_prompt_n + G;
        std::vector<int32_t> tokens((size_t) tok_cap, 0);
        for (int i = 0; i < stream_prompt_n; i++) tokens[i] = prompt_tok[i];
        for (int i = 0; i < G; i++) tokens[stream_prompt_n + i] = stream_flow_codes[begin + i];
        std::vector<float> token_mask((size_t) tok_cap, 0.0f);
        std::vector<float> mel_mask((size_t) mel_cap, 0.0f);
        std::fill_n(token_mask.begin(), total_tokens, 1.0f);
        std::fill_n(mel_mask.begin(), Tw, 1.0f);
        std::vector<float> pfeat((size_t) mel_cap * 80, 0.0f);
        const int pframes = std::min({(int) rf.flow_frames, Tw, mel_cap});
        memcpy(pfeat.data(), rf.flowfeat.data(), (size_t) pframes * 80 * sizeof(float));
        std::vector<float> noise((size_t) mel_cap * 80, 0.0f);
        std::vector<float> emb = rf.embedding;
        const int gen0 = 2 * stream_prompt_n;
        for (int c = 0; c < 80; c++) {
            for (int t = 0; t < Tw; t++) {
                const size_t noise_t = t < gen0
                        ? (size_t) t
                        : (size_t) gen0 + (size_t) 2 * begin + (t - gen0);
                noise[(size_t) c * mel_cap + t] = tab.flow_noise[(size_t) c * 15000 + noise_t];
            }
        }
        const auto tf0 = std::chrono::steady_clock::now();
        if (!flow_loop_npu->set_input("tokens", tokens.data(), tokens.size() * 4) ||
            !flow_loop_npu->set_input("token_mask", token_mask.data(), token_mask.size() * 4) ||
            !flow_loop_npu->set_input("mel_mask", mel_mask.data(), mel_mask.size() * 4) ||
            !flow_loop_npu->set_input("prompt_feat", pfeat.data(), pfeat.size() * 4) ||
            !flow_loop_npu->set_input("embedding", emb.data(), emb.size() * 4) ||
            !flow_loop_npu->set_input("noise", noise.data(), noise.size() * 4) ||
            (flow_loop_npu->input_elems("noise_state") > 0 &&
             !flow_loop_npu->set_input("noise_state", noise.data(), noise.size() * 4)) ||
            (flow_loop_npu->input_elems("noise_est") > 0 &&
             !flow_loop_npu->set_input("noise_est", noise.data(), noise.size() * 4)) ||
            (flow_loop_npu->input_elems("noise_est2") > 0 &&
             !flow_loop_npu->set_input("noise_est2", noise.data(), noise.size() * 4)) ||
            !flow_loop_npu->run()) {
            stream_error = "streaming flow kernel failed";
            return false;
        }
        const float * loop_out = flow_loop_npu->output_f32((size_t) 80 * mel_cap, flow_out_idx);
        if (!loop_out) {
            stream_error = "streaming flow produced no output";
            return false;
        }
        stream_flow_s += sec(tf0, std::chrono::steady_clock::now());
        std::vector<float> out_buf((size_t) Tw * 80);
        for (int c = 0; c < 80; c++) {
            for (int t = 0; t < Tw; t++) {
                float v = loop_out[(size_t) c * mel_cap + t];
                if (t >= gen0) v = flow_mel_value(tab, c, v);
                out_buf[(size_t) t * 80 + c] = v;
            }
        }
        const float * out = out_buf.data();
        const int old_frames = (int) stream_mel_ch[0].size();
        const int history_start = std::max(0, old_frames - a.hift_left);
        const int history = old_frames - history_start;
        const int dst_off = a.hift_left - history;

        const int cap = (int) (hift_npu->input_elems("mel") / 80);
        if (cap <= 64) {
            stream_error = "streaming HiFT kernel has an invalid window";
            return false;
        }
        const int s_frames = cap * 120 + 1;
        std::vector<float> front_mel((size_t) 80 * (history + new_frames));
        for (int c = 0; c < 80; c++) {
            if (history) memcpy(front_mel.data() + (size_t) c * (history + new_frames),
                                stream_mel_ch[c].data() + history_start, (size_t) history * 4);
            for (int i = 0; i < new_frames; i++)
                front_mel[(size_t) c * (history + new_frames) + history + i] =
                    out[(size_t) (gen0 + new_start + i) * 80 + c];
        }
        std::vector<float> hmel((size_t) 80 * cap, 0.0f);
        std::vector<float> s_chunk((size_t) 18 * s_frames, 0.0f);
        for (int c = 0; c < 80; c++) {
            if (history) memcpy(hmel.data() + (size_t) c * cap + dst_off,
                                stream_mel_ch[c].data() + history_start, (size_t) history * 4);
            for (int i = 0; i < new_frames; i++)
                hmel[(size_t) c * cap + 64 + i] = out[(size_t) (gen0 + new_start + i) * 80 + c];
        }
        // The f0 predictor now lives in the HiFT AXModel.  Run it on the same
        // window first (the decoder half of that pass is discarded), then feed
        // its f0 to the host source/STFT and run the decoder for real.
        std::vector<float> f0w;
        const int f0_idx = hift_npu->output_index("f0") >= 0 ? hift_npu->output_index("f0")
                                                  : hift_npu->output_index("f0_out");
        if (f0_idx >= 0) {
            const auto tz0 = std::chrono::steady_clock::now();
            if (!hift_npu->set_input("mel", hmel.data(), hmel.size() * 4) ||
                !hift_npu->set_input("s_stft", s_chunk.data(), s_chunk.size() * 4) || !hift_npu->run()) {
                stream_error = "streaming HiFT f0 pass failed";
                return false;
            }
            const float * f0p = hift_npu->output_f32(cap, f0_idx);
            if (!f0p) {
                stream_error = "streaming HiFT f0 output missing";
                return false;
            }
            f0w.assign(f0p, f0p + cap);
            stream_front_s += sec(tz0, std::chrono::steady_clock::now());
        }
        std::vector<float> s_front;
        int s_front_frames = 0;
        std::vector<double> s_phase((size_t) (history + new_frames) * 9);
        const double * phase0 = history_start ? stream_phase_frames.data() + (size_t) (history_start - 1) * 9
                                              : nullptr;
        const auto tfront0 = std::chrono::steady_clock::now();
        hift_front_native(native_hift, front_mel.data(), history + new_frames, s_front, s_front_frames,
                          phase0, nullptr, (size_t) history_start * 480, s_phase.data(),
                          f0w.empty() ? nullptr : f0w.data() + dst_off);
        stream_front_s += sec(tfront0, std::chrono::steady_clock::now());
        stream_phase_frames.resize((size_t) (old_frames + new_frames) * 9);
        memcpy(stream_phase_frames.data() + (size_t) old_frames * 9,
               s_phase.data() + (size_t) history * 9, (size_t) new_frames * 9 * sizeof(double));
        for (int c = 0; c < 18; c++) {
            memcpy(s_chunk.data() + (size_t) c * s_frames + (size_t) dst_off * 120,
                   s_front.data() + (size_t) c * s_front_frames, (size_t) s_front_frames * 4);
        }
        const auto th0 = std::chrono::steady_clock::now();
        if (!hift_npu->set_input("mel", hmel.data(), hmel.size() * 4) ||
            !hift_npu->set_input("s_stft", s_chunk.data(), s_chunk.size() * 4) || !hift_npu->run()) {
            stream_error = "streaming HiFT kernel failed";
            return false;
        }
        const size_t mp_bytes = hift_npu->output_bytes(0);
        const int mp_ch = mp_bytes ? (int) (mp_bytes / (sizeof(float) * (size_t) s_frames)) : 0;
        const float * mp = hift_npu->output_f32((size_t) mp_ch * s_frames, 0);
        if (!mp || mp_ch < 18) {
            stream_error = "streaming HiFT produced a bad output";
            return false;
        }
        std::vector<float> mp18;
        if (mp_ch > 18) {
            mp18.resize((size_t) 18 * s_frames);
            for (int c = 0; c < 18; c++) {
                memcpy(mp18.data() + (size_t) c * s_frames,
                       mp + (size_t) c * mp_ch * s_frames, (size_t) s_frames * 4);
            }
            mp = mp18.data();
        }
        stream_hift_s += sec(th0, std::chrono::steady_clock::now());
        const auto tp0 = std::chrono::steady_clock::now();
        std::vector<float> decoded;
        const int valid_frames = 64 + new_frames;
        hift_post_native(mp, valid_frames * 120 + 1, decoded, s_frames);
        if (decoded.size() < (size_t) (64 + new_frames) * 480) {
            stream_error = "streaming HiFT post produced too few samples";
            return false;
        }
        stream_post_s += sec(tp0, std::chrono::steady_clock::now());
        const size_t sample_off = (size_t) 64 * 480;
        const size_t n_samples = (size_t) new_frames * 480;
        std::vector<uint8_t> pcm(n_samples * 2);
        int16_t * dst = (int16_t *) pcm.data();
        for (size_t i = 0; i < n_samples; i++) {
            const float v = std::max(-1.0f, std::min(1.0f, decoded[sample_off + i]));
            dst[i] = (int16_t) std::lrint(v * 32767.0f);
        }
        if (!stream_cb(pcm, false)) {
            stream_error = "streaming client cancelled";
            return false;
        }
        if (stream_first_s < 0.0) {
            stream_first_s = sec(t_req0, std::chrono::steady_clock::now());
        }
        for (int c = 0; c < 80; c++)
            for (int i = 0; i < new_frames; i++)
                stream_mel_ch[c].push_back(out[(size_t) (gen0 + new_start + i) * 80 + c]);
        stream_hift_calls++;
        stream_out_codes = available;
        return true;
    };
    const float * hidden = nullptr;
    auto t_prefill = std::chrono::steady_clock::now();
    double t_prefill_graph = 0.0;
    if (codes_path.empty() && !prefix_hit) {
        const auto t_pg0 = std::chrono::steady_clock::now();
        if (decode_embd(ctx, batch.batch, lm_input.data(), n_prompt_rows, n_embd, 0, true) != 0) {
            fprintf(stderr, "prefill decode failed\n"); return 1;
        }
        t_prefill_graph = sec(t_pg0, std::chrono::steady_clock::now());
        hidden = llama_get_embeddings_ith(ctx, -1);
        if (!hidden) { fprintf(stderr, "no hidden state\n"); return 1; }
        h.assign(hidden, hidden + n_embd);
        t_prefill = std::chrono::steady_clock::now();
    } else if (prefix_hit) {
        t_prefill = std::chrono::steady_clock::now();
    }

    // ---- autoregressive speech token sampling ---------------------------
    // The speech decoder is the output head of both stage AXModels:
    // llama_get_logits_ith returns its 6761 real rows (the padded tail is -inf).
    std::vector<float> logits(aux.n_speech);
    auto read_logits = [&](std::vector<float> & out) {
        const float * l = llama_get_logits_ith(ctx, -1);
        if (!l) {
            return false;
        }
        std::copy(l, l + aux.n_speech, out.begin());
        return true;
    };
    double t_logits_initial = 0.0, t_cache_cp = 0.0;
    if (codes_path.empty()) {
        if (prefix_hit) {
            logits = pcache[prefix_hit_idx].logits;
            fprintf(stderr, "[llm] prefix cache hit rows=%d\n", n_prompt_rows);
        } else {
            const auto t_l0 = std::chrono::steady_clock::now();
            if (!read_logits(logits)) {
                fprintf(stderr, "no speech logits from the prefill graph\n");
                return 1;
            }
            t_logits_initial = sec(t_l0, std::chrono::steady_clock::now());
            const auto t_c0 = std::chrono::steady_clock::now();
            prompt_kv_cache entry;
            entry.rows = n_prompt_rows;
            entry.ids = prefix_key;
            entry.logits = logits;
            const size_t state_size = llama_state_seq_get_size(ctx, 0);
            if (state_size > 0) {
                entry.state.resize(state_size);
                const size_t got = llama_state_seq_get_data(
                        ctx, entry.state.data(), entry.state.size(), 0);
                if (got != entry.state.size()) entry.state.clear();
            }
            if (!entry.state.empty()) {
                entry.used = ++pcache_tick;
                pcache.push_back(std::move(entry));
                if (pcache.size() > PCACHE_MAX) {
                    auto oldest = std::min_element(pcache.begin(), pcache.end(),
                            [](const auto & a, const auto & b) { return a.used < b.used; });
                    pcache.erase(oldest);
                }
            }
            t_cache_cp = sec(t_c0, std::chrono::steady_clock::now());
        }
    }
    if (codes_path.empty()) {
        fprintf(stderr,
                "[llm-host2] tokenize=%.4f embed=%.4f kv_rm=%.4f kv_hit=%.4f prefill_graph=%.4f "
                "logits_prefill=%.4f cache_cp=%.4f rows=%d prefix_hit=%d\n",
                t_tokenize, t_embed, t_kv_rm, t_kv_hit, t_prefill_graph,
                t_logits_initial, t_cache_cp, n_prompt_rows, prefix_hit ? 1 : 0);
    }
    std::vector<int> codes;
    auto t_gen0 = std::chrono::steady_clock::now();
    double t_sampling = 0.0, t_decode = 0.0, t_logits_copy = 0.0;
    if (!codes_path.empty()) {
        if (!load_codes(codes_path, codes)) return 1;
        fprintf(stderr, "[llm] injected %zu speech tokens from %s\n", codes.size(), codes_path.c_str());
    } else {
        sampler smp(req_seed, a.greedy);
        int max_len = (int) text_tok.size() * 20;
        if (flow_loop_npu) {
            // The flow window is reused for every chunk of a long sentence, so
            // the decoded length is bounded by the text, not by the window.
            const int tok_cap = (int) flow_loop_npu->input_elems("tokens");
            max_len = std::max(max_len, tok_cap);
        }
        const int min_len = std::min((int) text_tok.size() * 2, max_len);
        int n_past = n_prompt_rows;
        for (int i = 0; i < max_len; i++) {
            auto ts0 = std::chrono::steady_clock::now();
            int tok = smp.sample(logits, codes, i < min_len, 6561);
            t_sampling += sec(ts0, std::chrono::steady_clock::now());
            if (i >= min_len && tok >= 6561) break;
            if (tok < 0 || tok >= 6561) break;
            codes.push_back(tok);
            if (streaming) {
                if (!flow_silent_token(tok, stream_silent_run)) stream_flow_codes.push_back(tok);
                const int pending = (int) stream_flow_codes.size() - stream_out_codes;
                const int threshold = stream_out_codes == 0 ? 2 : std::min(64, stream_chunk_cap);
                const bool full = pending >= stream_chunk_cap;
                if ((pending >= threshold || full) && !run_stream_window()) {
                    fprintf(stderr, "streaming synthesis failed: %s\n", stream_error.c_str());
                    return 1;
                }
            }
            if (i + 1 >= max_len) break;
            auto td0 = std::chrono::steady_clock::now();
            if (decode_embd(ctx, batch.batch, speech_row(tok), 1, n_embd, n_past, true) != 0) {
                fprintf(stderr, "decode step %d failed\n", i); return 1;
            }
            t_decode += sec(td0, std::chrono::steady_clock::now());
            n_past++;
            auto tl0 = std::chrono::steady_clock::now();
            if (!read_logits(logits)) {
                fprintf(stderr, "no speech logits at step %d\n", i); return 1;
            }
            t_logits_copy += sec(tl0, std::chrono::steady_clock::now());
        }
    }
    if (streaming) {
        while (stream_out_codes < (int) stream_flow_codes.size()) {
            if (!run_stream_window()) {
                fprintf(stderr, "streaming synthesis failed: %s\n", stream_error.c_str());
                return 1;
            }
        }
    }
    if (streaming && std::getenv("COSY_STREAM_DUMP")) {
        const int Tn = (int) stream_mel_ch[0].size();
        FILE * f = fopen("/tmp/stream_mel.f32", "wb");
        if (f) {
            for (int c = 0; c < 80; c++) fwrite(stream_mel_ch[c].data(), 4, Tn, f);
            fclose(f);
        }
        f = fopen("/tmp/stream_codes.bin", "wb");
        if (f) {
            for (int v : stream_flow_codes) { int32_t w = v; fwrite(&w, 4, 1, f); }
            fclose(f);
        }
    }
    auto t_gen1 = std::chrono::steady_clock::now();
    if (codes.empty()) { fprintf(stderr, "no speech tokens generated\n"); return 1; }
    fprintf(stderr, "[llm] generated %zu speech tokens\n", codes.size());
    if (codes_path.empty()) {
        const double gen = sec(t_gen0, t_gen1);
        fprintf(stderr, "[llm-host] sampling %.3fs decode_graph %.3fs logits_copy %.3fs other %.3fs total %.3fs\n",
                t_sampling, t_decode, t_logits_copy,
                std::max(0.0, gen - t_sampling - t_decode - t_logits_copy), gen);
    }

    if (a.dump_codes) {
        std::string path = a.out + ".codes.bin";
        FILE * f = fopen(path.c_str(), "wb");
        for (int c : codes) { int32_t v = c; fwrite(&v, 4, 1, f); }
        fclose(f);
        fprintf(stderr, "[llm] wrote %s\n", path.c_str());
    }

    // ---- flow encoder ---------------------------------------------------
    std::vector<int> flow_codes = streaming ? stream_flow_codes : filter_flow_silent_tokens(codes);
    if (flow_codes.size() != codes.size()) {
        fprintf(stderr, "[flow] filtered %zu silent speech tokens (%zu -> %zu)\n",
                codes.size() - flow_codes.size(), codes.size(), flow_codes.size());
    }
    std::vector<int32_t> gen_tok(flow_codes.begin(), flow_codes.end());
    std::vector<float> mel;
    int Tgen = 0, T_total = 0;
    std::chrono::steady_clock::time_point t_flowenc, t_flow;
    if (streaming) {
        t_flowenc = std::chrono::steady_clock::now();
        Tgen = (int) stream_mel_ch[0].size();
        mel.assign((size_t) 80 * Tgen, 0.0f);
        for (int c = 0; c < 80; c++) {
            memcpy(mel.data() + (size_t) c * Tgen, stream_mel_ch[c].data(), (size_t) Tgen * 4);
        }
        T_total = 2 * (int) prompt_tok.size() + Tgen;
        t_flow = std::chrono::steady_clock::now();
    } else if (flow_loop_npu) {
        t_flowenc = std::chrono::steady_clock::now();
        const int tok_cap = (int) flow_loop_npu->input_elems("tokens");
        const int mel_cap = (int) (flow_loop_npu->input_elems("noise") / 80);
        const int noise_cap = 15000;
        if (tok_cap <= 0 || mel_cap != 2 * tok_cap) {
            fprintf(stderr, "flow loop has unexpected capacities tokens=%d mel=%d\n", tok_cap, mel_cap);
            return 1;
        }
        const int full_prompt_n = (int) prompt_tok.size();
        // Keep the longest reference prefix that leaves a useful chunk, with
        // --flow-prompt-min as the lower bound.  The speaker embedding and the
        // language-model prompt remain complete.
        const int prompt_floor = std::min(full_prompt_n, std::max(8, a.flow_prompt_min));
        const int n_codes = (int) flow_codes.size();
        // Keep the whole reference prefix when the generated codes leave room.
        // When they do not, trim only as much of the prompt as needed so this
        // segment still fits in one flow window, never below --flow-prompt-min.
        const int prompt_need = tok_cap - n_codes - 4;
        int prompt_n = full_prompt_n;
        if (prompt_need < full_prompt_n) {
            prompt_n = std::min(full_prompt_n, std::max(prompt_floor, prompt_need));
        }
        const int prompt_mel = 2 * prompt_n;
        if (full_prompt_n > prompt_n) {
            fprintf(stderr, "[flow] reference prompt %d tokens truncated to %d for the merged window\n",
                    full_prompt_n, prompt_n);
        }
        const int chunk_cap = std::max(1, tok_cap - prompt_n - 4);
        // An independent window cannot continue the previous one, so keep a
        // few tokens of left context and drop their duplicated mel frames.
        const int overlap = std::min(4, chunk_cap / 4);
        Tgen = 2 * n_codes;
        T_total = prompt_mel + Tgen;
        mel.assign((size_t) 80 * Tgen, 0.0f);
        std::vector<int32_t> tokens(tok_cap, 0);
        std::vector<float> pfeat((size_t) mel_cap * 80, 0.0f);
        const int pframes = std::min({(int) rf.flow_frames, prompt_mel, mel_cap});
        memcpy(pfeat.data(), rf.flowfeat.data(), (size_t) pframes * 80 * sizeof(float));
        std::vector<float> noise((size_t) 80 * mel_cap, 0.0f);
        std::vector<float> emb = rf.embedding;
        if ((int) emb.size() != 192) { fprintf(stderr, "flow embedding has %zu values\n", emb.size()); return 1; }
        const bool mask_inputs = flow_loop_npu->input_elems("token_mask") > 0;
        std::vector<float> token_mask(mask_inputs ? tok_cap : 0, 0.0f);
        std::vector<float> mel_mask(mask_inputs ? mel_cap : 0, 0.0f);
        for (int c = 0; c < 80; c++) {
            const float * src = tab.flow_noise.data() + (size_t) c * noise_cap;
            memcpy(noise.data() + (size_t) c * mel_cap, src, (size_t) mel_cap * sizeof(float));
        }
        int windows = 0;
        for (int start = 0; start < n_codes; ) {
            const int ctx = start == 0 ? 0 : std::min(overlap, start);
            const int begin = start - ctx;
            const int end = std::min(n_codes, begin + chunk_cap);
            const int window_codes = end - begin;
            const int new_codes = end - start;
            if (window_codes <= 0 || new_codes <= 0) {
                fprintf(stderr, "[flow] chunking stalled at %d/%d\n", start, n_codes);
                return 1;
            }
            const int total_tokens = prompt_n + window_codes;
            std::fill(tokens.begin(), tokens.end(), 0);
            for (int i = 0; i < prompt_n; i++) tokens[i] = prompt_tok[i];
            for (int i = 0; i < window_codes; i++) tokens[prompt_n + i] = flow_codes[begin + i];
            if (mask_inputs) {
                std::fill(token_mask.begin(), token_mask.end(), 0.0f);
                std::fill(mel_mask.begin(), mel_mask.end(), 0.0f);
                std::fill_n(token_mask.begin(), total_tokens, 1.0f);
                std::fill_n(mel_mask.begin(), 2 * total_tokens, 1.0f);
            }
            const int32_t gen_len_i = (int32_t) window_codes;
            const int32_t prompt_len_i = prompt_n;
            bool flow_ok = flow_loop_npu->set_input("tokens", tokens.data(), tokens.size() * 4);
            if (flow_ok && mask_inputs) {
                flow_ok = flow_loop_npu->set_input("token_mask", token_mask.data(), token_mask.size() * 4) &&
                          flow_loop_npu->set_input("mel_mask", mel_mask.data(), mel_mask.size() * 4);
            } else if (flow_ok) {
                flow_ok = flow_loop_npu->set_input("gen_len", &gen_len_i, 4) &&
                          flow_loop_npu->set_input("prompt_len", &prompt_len_i, 4);
            }
            flow_ok = flow_ok &&
                      flow_loop_npu->set_input("prompt_feat", pfeat.data(), pfeat.size() * 4) &&
                      flow_loop_npu->set_input("embedding", emb.data(), emb.size() * 4) &&
                      flow_loop_npu->set_input("noise", noise.data(), noise.size() * 4) &&
                      (flow_loop_npu->input_elems("noise_state") <= 0 ||
                       flow_loop_npu->set_input("noise_state", noise.data(), noise.size() * 4)) &&
                      (flow_loop_npu->input_elems("noise_est") <= 0 ||
                       flow_loop_npu->set_input("noise_est", noise.data(), noise.size() * 4)) &&
                      (flow_loop_npu->input_elems("noise_est2") <= 0 ||
                       flow_loop_npu->set_input("noise_est2", noise.data(), noise.size() * 4)) &&
                      flow_loop_npu->run();
            if (!flow_ok) return 1;
            const float * out = flow_loop_npu->output_f32((size_t) 80 * mel_cap, flow_out_idx);
            if (!out) return 1;
            const int src_frame = prompt_mel + 2 * ctx;
            const int dst_frame = 2 * start;
            const int frames = 2 * new_codes;
            for (int c = 0; c < 80; c++) {
                for (int t = 0; t < frames; t++) {
                    mel[(size_t) c * Tgen + dst_frame + t] =
                            flow_mel_value(tab, c, out[(size_t) c * mel_cap + src_frame + t]);
                }
            }
            fprintf(stderr, "[flow-win] pn=%d ctx=%d new=%d total=%d cap=%d\n",
                    prompt_n, ctx, new_codes, total_tokens, tok_cap);
            start = end;
            windows++;
        }
        fprintf(stderr, "[flow] windows=%d cap_tokens=%d cap_mel=%d\n", windows, tok_cap, mel_cap);
        t_flow = std::chrono::steady_clock::now();
    } else {
        fprintf(stderr, "the merged flow AXModel is required\n");
        return 1;
    }
    if (!a.dump_mel.empty()) {
        FILE * f = fopen(a.dump_mel.c_str(), "wb");
        if (!f) { fprintf(stderr, "cannot write %s\n", a.dump_mel.c_str()); return 1; }
        fwrite(mel.data(), 4, mel.size(), f);
        fclose(f);
        fprintf(stderr, "[flow] wrote %s (%d x %d)\n", a.dump_mel.c_str(), 80, Tgen);
    }
    if (a.mel_only) {
        fprintf(stderr,
                "[timing] frontend %.3fs llm_load %.3fs prefill %.3fs gen %.3fs (%.1f tok/s) flow_enc %.3fs flow %.3fs hift %.3fs total %.3fs\n",
                front_s, llm_load_s, sec(t_req0, t_prefill), sec(t_gen0, t_gen1),
                codes.size() / std::max(1e-9, sec(t_gen0, t_gen1)),
                sec(t_gen1, t_flowenc), sec(t_flowenc, t_flow), 0.0,
                front_s + llm_load_s + sec(t_req0, t_flow));
        return 0;
    }
    std::vector<float> pcm;
    int hift_calls = 0;
    if (streaming) {
        pcm.resize((size_t) Tgen * 480);
    } else if (hift_npu) {
        npu_model * hmodel = hift_npu.get();
        const int cap = (int) (hift_npu->input_elems("mel") / 80);
        // Reuse the one 512-frame vocoder for long utterances.  CausalHiFT is
        // convolution-only after the CPU front end.  Compute the source STFT
        // once on the full mel (so SineGen's phase is continuous), then give
        // each decoder window 64 frames of left context.
        const int left = a.hift_left;
        // conv_pre is right-causal and right-pads four mel frames, so every
        // window must also carry four real future frames after its valid span.
        constexpr int right = 4;
        const int stride = cap - left - right;
        const bool native = have_native_hift;
        const bool split = native;
        if (cap <= 64) {
            fprintf(stderr, "HiFT kernel has an invalid window %d\n", cap);
            return 1;
        }
        const int s_frames = cap * 480 / 4 + 1;
        double t_front = 0.0, t_npu = 0.0, t_post = 0.0, t_f0npu = 0.0;
        int chunks = 0;
        std::vector<float> full_s;
        std::vector<float> f0_all;
        const int hift_f0_idx = !split ? -1 :
            (hmodel->output_index("f0") >= 0 ? hmodel->output_index("f0") : hmodel->output_index("f0_out"));
        if (hift_f0_idx >= 0) {
            // The f0 predictor is part of the HiFT AXModel.  Sweep the same
            // decoder windows once with a zero s_stft input to obtain f0 for
            // the whole utterance; the host then only builds the sine source
            // and its STFT.  Phase continuity stays exactly as before because
            // the source runs once over the full mel.
            auto tz0 = std::chrono::steady_clock::now();
            f0_all.assign(Tgen, 0.0f);
            std::vector<float> s_zero((size_t) 18 * s_frames, 0.0f);
            std::vector<float> hmel((size_t) 80 * cap);
            for (int start = 0; start < Tgen; start += stride) {
                const int in_start = std::max(0, start - left);
                const int dst_off = start == 0 ? left : 0;
                const int take = std::min(cap - dst_off, Tgen - in_start);
                std::fill(hmel.begin(), hmel.end(), 0.0f);
                for (int c = 0; c < 80; c++) {
                    memcpy(hmel.data() + (size_t) c * cap + dst_off,
                           mel.data() + (size_t) c * Tgen + in_start, (size_t) take * 4);
                }
                if (!hmodel->set_input("mel", hmel.data(), hmel.size() * 4) ||
                    !hmodel->set_input("s_stft", s_zero.data(), s_zero.size() * 4) || !hmodel->run()) {
                    fprintf(stderr, "hift f0 sweep failed\n");
                    return 1;
                }
                const float * f0p = hmodel->output_f32(cap, hift_f0_idx);
                if (!f0p) return 1;
                memcpy(f0_all.data() + in_start, f0p + dst_off, (size_t) take * 4);
            }
            t_f0npu = sec(tz0, std::chrono::steady_clock::now());
        }
        if (split) {
            auto t0 = std::chrono::steady_clock::now();
            int nf_native = 0;
            hift_front_native(native_hift, mel.data(), Tgen, full_s, nf_native,
                              nullptr, nullptr, 0, nullptr, f0_all.empty() ? nullptr : f0_all.data());
            if (nf_native != Tgen * 120 + 1 || full_s.size() != (size_t) 18 * nf_native) {
                fprintf(stderr, "native hift front produced a bad shape for %d frames\n", Tgen);
                return 1;
            }
            if (!a.dump_hift_s.empty()) {
                FILE * f = fopen(a.dump_hift_s.c_str(), "wb");
                if (f) { fwrite(full_s.data(), sizeof(float), full_s.size(), f); fclose(f); }
            }
            t_front = sec(t0, std::chrono::steady_clock::now());
        }
        std::vector<float> audio_out((size_t) Tgen * 480);
        for (int start = 0; start < Tgen; start += stride) {
            const int in_start = std::max(0, start - left);
            // The first window starts the graph at row zero.  HiFT's tiled
            // convolution path is not numerically stable there because it has
            // no real left context.  Reserve the same 64-frame history used by
            // every later window and prepend zeros instead; the discarded
            // prefix then also moves the first consumed row off the boundary.
            const int dst_off = split && start == 0 ? left : 0;
            const int off = start - in_start + dst_off;
            const int valid = std::min(stride, Tgen - start);
            const int take = std::min(cap - dst_off, Tgen - in_start);
            std::vector<float> hmel((size_t) 80 * cap, 0.0f);
            for (int c = 0; c < 80; c++) {
                memcpy(hmel.data() + (size_t) c * cap + dst_off,
                       mel.data() + (size_t) c * Tgen + in_start, (size_t) take * 4);
            }
            const float * audio = nullptr;
            std::vector<float> decoded;
            if (split) {
                auto th1 = std::chrono::steady_clock::now();
                const size_t s0 = (size_t) in_start * 120;
                const size_t s_dst = (size_t) dst_off * 120;
                const size_t s_take = std::min((size_t) s_frames - s_dst,
                                               (size_t) Tgen * 120 + 1 - s0);
                std::vector<float> s_chunk((size_t) 18 * s_frames, 0.0f);
                for (int c = 0; c < 18; c++) {
                    memcpy(s_chunk.data() + (size_t) c * s_frames + s_dst,
                           full_s.data() + (size_t) c * (Tgen * 120 + 1) + s0, s_take * 4);
                }
                if (!hmodel->set_input("mel", hmel.data(), hmel.size() * 4) ||
                    !hmodel->set_input("s_stft", s_chunk.data(), s_chunk.size() * 4) || !hmodel->run()) return 1;
                const size_t mp_bytes = hmodel->output_bytes();
                if (!mp_bytes || mp_bytes % (sizeof(float) * (size_t) s_frames) != 0) {
                    fprintf(stderr, "hift NPU output has an unexpected size %zu\n", mp_bytes);
                    return 1;
                }
                const int mp_ch = (int) (mp_bytes / (sizeof(float) * (size_t) s_frames));
                const float * mp = hmodel->output_f32((size_t) mp_ch * s_frames);
                if (!mp) return 1;
                std::vector<float> mp18;
                if (mp_ch > 18) {
                    mp18.resize((size_t) 18 * s_frames);
                    for (int c = 0; c < 18; c++) {
                        memcpy(mp18.data() + (size_t) c * s_frames,
                               mp + (size_t) c * mp_ch * s_frames, (size_t) s_frames * 4);
                    }
                    mp = mp18.data();
                } else if (mp_ch != 18) {
                    fprintf(stderr, "hift NPU output has %d channels\n", mp_ch);
                    return 1;
                }
                if (!a.dump_hift_mp.empty()) {
                    char name[1024];
                    snprintf(name, sizeof(name), "%s.%d.f32", a.dump_hift_mp.c_str(), chunks);
                    FILE * f = fopen(name, "wb");
                    if (f) { fwrite(mp, sizeof(float), (size_t) 18 * s_frames, f); fclose(f); }
                }
                auto th2 = std::chrono::steady_clock::now();
                if (native) {
                    const int valid_frames = std::min(cap, dst_off + take);
                    hift_post_native(mp, valid_frames * 120 + 1, decoded, s_frames);
                    if (decoded.size() < (size_t) valid_frames * 480) {
                        fprintf(stderr, "native hift post produced %zu samples, expected %d\\n",
                                decoded.size(), valid_frames * 480);
                        return 1;
                    }
                    decoded.resize((size_t) valid_frames * 480);
                }
                audio = decoded.data();
                auto th3 = std::chrono::steady_clock::now();
                t_npu += sec(th1, th2);
                t_post += sec(th2, th3);
            }
            memcpy(audio_out.data() + (size_t) start * 480, audio + (size_t) off * 480, (size_t) valid * 480 * 4);
            chunks++;
        }
        pcm.swap(audio_out);
        if (pcm.size() != (size_t) Tgen * 480) {
            fprintf(stderr, "hift produced %zu samples, expected %zu\n", pcm.size(), (size_t) Tgen * 480);
            return 1;
        }
        fprintf(stderr, "[hift] chunks=%d cap=%d front=%.3fs f0npu=%.3fs npu=%.3fs post=%.3fs\n",
                chunks, cap, t_front, t_f0npu, t_npu, t_post);
        hift_calls = chunks;
    } else {
        fprintf(stderr, "the HiFT AXModel is required\n");
        return 1;
    }
    auto t_hift = std::chrono::steady_clock::now();
    if (streaming) {
        hift_calls = stream_hift_calls;
    }

    if (!streaming && !write_wav(outpath.c_str(), pcm, 24000)) {
        fprintf(stderr, "failed to write %s\n", outpath.c_str());
        return 1;
    }

    fprintf(stderr,
            "[timing] frontend %.3fs llm_load %.3fs prefill %.3fs gen %.3fs (%.1f tok/s) flow_enc %.3fs flow %.3fs hift %.3fs total %.3fs\n",
            front_s, llm_load_s,
            sec(t_req0, t_prefill), sec(t_gen0, t_gen1), codes.size() / std::max(1e-9, sec(t_gen0, t_gen1)),
            sec(t_gen1, t_flowenc), sec(t_flowenc, t_flow), sec(t_flow, t_hift),
            front_s + llm_load_s + sec(t_req0, t_hift));
    fprintf(stderr, "[audio] %.3fs %zu samples, codes=%zu flow_codes=%zu, T=%d Tgen=%d\n",
            pcm.size() / 24000.0, pcm.size(), codes.size(), flow_codes.size(), T_total, Tgen);
    if (streaming) {
        fprintf(stderr,
                "[stream] first=%.3fs chunks=%d flow_windows=%d flow=%.3fs front=%.3fs hift=%.3fs post=%.3fs\n",
                stream_first_s, stream_hift_calls, stream_hift_calls,
                stream_flow_s, stream_front_s,
                stream_hift_s, stream_post_s);
    }
    fprintf(stderr,
            "[perf] prefill=%.3f gen=%.3f flow=%.3f hift=%.3f front_cache=%.3f codes=%zu "
            "flow_calls=%d hift_calls=%d streaming=%d\n",
            sec(t_req0, t_prefill), sec(t_gen0, t_gen1),
            streaming ? stream_flow_s : sec(t_flowenc, t_flow),
            streaming ? (stream_front_s + stream_hift_s + stream_post_s) : sec(t_flow, t_hift),
            front_s, codes.size(),
            streaming ? stream_hift_calls : 1, hift_calls, streaming ? 1 : 0);
    fprintf(stderr, "[timing2] aux %.3fs sessions %.3fs npu_load %.3fs request %.3fs wall %.3fs\n",
            sec(t_aux0, t_aux1), t_sess, t_npu_load, front_s + llm_load_s + sec(t_req0, t_hift),
            sec(t_all0, std::chrono::steady_clock::now()));
    fprintf(stderr, "[out] %s\n", outpath.c_str());
    return 0;
    };

    // A single fixed flow window cannot consume an arbitrarily long
    // utterance.  Split at punctuation and run each piece as its own
    // zero-shot sentence with the same reference; concatenate the PCM for a
    // non-stream request and let the streaming callback deliver each piece.
    auto synthesize = [&](const std::string & req_text, const std::string & req_prompt_text,
                          const std::string & outpath, const ref_feats & rf, uint32_t req_seed,
                          const std::string & codes_path, double front_s, double llm_load_s,
                          bool incremental, const cosy_audio_chunk_callback & stream_cb) -> int {
        std::vector<std::string> segments{req_text};
        int gen_cap = 0;
        if (codes_path.empty() && flow_loop_npu) {
            const int tok_cap = (int) flow_loop_npu->input_elems("tokens");
            const int prompt_floor = std::min((int) rf.tok.size(), std::max(8, a.flow_prompt_min));
            gen_cap = tok_cap - prompt_floor - 4;
            segments = split_text_segments(req_text, gen_cap);
        }
        const auto req_t0 = std::chrono::steady_clock::now();
        fprintf(stderr, "[request] begin segments=%zu chars=%zu gen_cap=%d\n",
                segments.size(), req_text.size(), gen_cap);
        if (segments.size() > 1) {
            fprintf(stderr, "[segment] %zu segments, flow gen cap=%d\n", segments.size(), gen_cap);
        }
        int rc = 0;
        if (segments.size() <= 1) {
            rc = generate(req_text, req_prompt_text, outpath, rf, req_seed, codes_path,
                          front_s, llm_load_s, incremental, stream_cb);
        } else if (incremental) {
            for (size_t i = 0; i < segments.size() && rc == 0; i++) {
                fprintf(stderr, "[segment] %zu/%zu: %s\n", i + 1, segments.size(), segments[i].c_str());
                rc = generate(segments[i], req_prompt_text, outpath, rf, req_seed, std::string(),
                              0.0, 0.0, true, stream_cb);
            }
        } else {
            std::vector<std::string> parts;
            for (size_t i = 0; i < segments.size() && rc == 0; i++) {
                const std::string part = outpath + ".seg" + std::to_string(i) + ".wav";
                fprintf(stderr, "[segment] %zu/%zu: %s\n", i + 1, segments.size(), segments[i].c_str());
                rc = generate(segments[i], req_prompt_text, part, rf, req_seed, std::string(),
                              0.0, 0.0, false, {});
                if (rc == 0) parts.push_back(part);
            }
            if (rc == 0 && !concat_wav_parts(parts, outpath)) {
                fprintf(stderr, "failed to concatenate %zu segments into %s\n", parts.size(), outpath.c_str());
                rc = 1;
            }
            for (const auto & p : parts) unlink(p.c_str());
        }
        fprintf(stderr, "[request] end segments=%zu stream=%d error=%d wall=%.3f\n",
                segments.size(), incremental ? 1 : 0, rc != 0,
                sec(req_t0, std::chrono::steady_clock::now()));
        return rc;
    };

    if (service_mode) {
        const std::string warm_path = !a.warm_prompt.empty() ? a.warm_prompt : a.prompt_wav;
        const std::string warm_text = !a.warm_text.empty() ? a.warm_text : a.prompt_text;
        if (!warm_path.empty()) {
            auto warm = prepare_ref(warm_path);
            if (warm) {
                const std::string warm_out = "/tmp/cosy_service_warmup.wav";
                generate("你好。", warm_text, warm_out, *warm, a.seed, std::string(), 0.0, 0.0, false, {});
                unlink(warm_out.c_str());
            }
        }
        // A second bundled reference is a common first request (the shipped
        // English prompt).  Its speaker/tokenizer features and ONNX sessions
        // are built here so /health only becomes ready with both refs warm.
        if (!a.warm_prompt2.empty()) {
            auto warm2 = prepare_ref(a.warm_prompt2);
            if (warm2) {
                const std::string warm_out = "/tmp/cosy_service_warmup2.wav";
                const std::string t2 = a.warm_text2.empty() ? warm_text : a.warm_text2;
                generate("Hello.", t2, warm_out, *warm2, a.seed, std::string(), 0.0, 0.0, false, {});
                unlink(warm_out.c_str());
            }
        }
        {
            std::lock_guard<std::mutex> lock(g_service.mu);
            g_service.ready = true;
            g_service.starting = false;
        }
        g_service.cv.notify_all();

        uint64_t seq = 0;
        for (;;) {
            std::unique_ptr<cosy_service_request> req;
            {
                std::unique_lock<std::mutex> lock(g_service.mu);
                // A completed request stays in the slot until its caller has
                // copied the result and released the slot.  Without the
                // !done check the worker immediately re-takes the same object
                // while the caller is still reading it.
                g_service.cv.wait(lock, [&] {
                    return g_service.stopping ||
                           (g_service.request != nullptr && !g_service.request->done);
                });
                if (g_service.stopping) break;
                req = std::move(g_service.request);
            }
            const std::string tmp_ref = "/tmp/cosy_ref_" + std::to_string((long) getpid()) + "_" +
                                        std::to_string(seq++) + ".wav";
            const std::string tmp_out = "/tmp/cosy_out_" + std::to_string((long) getpid()) + "_" +
                                        std::to_string(seq) + ".wav";
            req->ok = false;
            if (!write_binary_file(tmp_ref, req->ref_wav)) {
                req->error = "failed to materialize reference audio";
            } else {
                auto t_front0 = std::chrono::steady_clock::now();
                auto rf = prepare_ref(tmp_ref);
                const double front_s = sec(t_front0, std::chrono::steady_clock::now());
                const int rc = rf ? synthesize(req->text, req->ref_text, tmp_out, *rf, req->seed,
                                               std::string(), front_s, 0.0, req->streaming, req->on_chunk) : 1;
                if (rc != 0) {
                    req->error = "CosyVoice3 synthesis failed";
                } else if (req->streaming) {
                    req->ok = true;
                } else {
                    std::vector<uint8_t> wav;
                    if (!read_binary_file(tmp_out, wav) || wav.size() <= 44) {
                        req->error = "CosyVoice3 produced no output";
                    } else {
                        // generate() wrote a RIFF file; llama-server adds the
                        // response header itself, so hand it raw PCM only.
                        req->result.assign(wav.begin() + 44, wav.end());
                        req->ok = true;
                    }
                }
            }
            unlink(tmp_ref.c_str());
            unlink(tmp_out.c_str());
            {
                std::lock_guard<std::mutex> lock(g_service.mu);
                req->done = true;
                g_service.request = std::move(req);
            }
            g_service.cv.notify_all();
        }
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    auto t_ref0 = std::chrono::steady_clock::now();
    auto rf0 = prepare_ref(a.prompt_wav);
    if (!rf0) return 1;
    double front_s = sec(t_ref0, std::chrono::steady_clock::now());
    double llm_load_s = sec(t_load0, t_load1);
    for (int rep = 0; rep < a.repeat; rep++) {
        std::string outpath = a.repeat > 1 && rep + 1 < a.repeat
            ? a.out + ".warm" + std::to_string(rep + 1) + ".wav" : a.out;
        if (synthesize(a.text, a.prompt_text, outpath, *rf0, a.seed,
                       rep == 0 ? a.codes_in : std::string(),
                       rep == 0 ? front_s : 0.0, rep == 0 ? llm_load_s : 0.0, false, {}) != 0) return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}

int cosyvoice3_cli_main(int argc, char ** argv) {
    return cosy_tts_main_impl(argc, argv, false);
}

bool cosyvoice3_service_start(const std::vector<std::string> & args, std::string & error) {
    std::unique_lock<std::mutex> lock(g_service.mu);
    if (g_service.thread.joinable()) {
        if (g_service.ready) return true;
        error = "CosyVoice3 service is already starting";
        return false;
    }
    g_service.ready = false;
    g_service.starting = true;
    g_service.failed = false;
    g_service.stopping = false;
    g_service.start_error.clear();
    g_service.request.reset();
    std::vector<std::string> owned = args;
    g_service.thread = std::thread([owned = std::move(owned)]() mutable {
        std::vector<char *> argv;
        argv.reserve(owned.size() + 1);
        for (auto & s : owned) argv.push_back(const_cast<char *>(s.c_str()));
        argv.push_back(nullptr);
        const int rc = cosy_tts_main_impl((int) owned.size(), argv.data(), true);
        std::lock_guard<std::mutex> lock(g_service.mu);
        if (!g_service.ready && !g_service.stopping) {
            g_service.failed = true;
            g_service.start_error = rc == 0 ? "CosyVoice3 service exited during initialization"
                                            : "CosyVoice3 service initialization failed";
        }
        g_service.starting = false;
        g_service.cv.notify_all();
    });
    g_service.cv.wait(lock, [&] { return g_service.ready || g_service.failed; });
    if (!g_service.ready) {
        error = g_service.start_error.empty() ? "CosyVoice3 service initialization failed" : g_service.start_error;
        if (g_service.thread.joinable()) {
            g_service.stopping = true;
            g_service.cv.notify_all();
        }
        return false;
    }
    return true;
}

bool cosyvoice3_service_request(const std::string & text,
                                const std::string & ref_text,
                                const std::vector<uint8_t> & ref_wav,
                                uint32_t seed, bool streaming,
                                const cosy_audio_chunk_callback & on_chunk,
                                std::string & error, std::string * pcm_out) {
    std::unique_lock<std::mutex> lock(g_service.mu);
    if (!g_service.ready || g_service.failed || g_service.stopping) {
        error = "CosyVoice3 service is not ready";
        return false;
    }
    g_service.cv.wait(lock, [&] { return g_service.request == nullptr || g_service.stopping; });
    if (g_service.stopping) {
        error = "CosyVoice3 service is stopping";
        return false;
    }
    auto req = std::make_unique<cosy_service_request>();
    req->text = text;
    req->ref_text = ref_text;
    req->ref_wav = ref_wav;
    req->seed = seed;
    req->streaming = streaming;
    req->on_chunk = on_chunk;
    auto * current = req.get();
    g_service.request = std::move(req);
    g_service.cv.notify_all();
    g_service.cv.wait(lock, [&] { return current->done || g_service.stopping; });
    if (!current->done) {
        error = "CosyVoice3 service stopped during request";
        return false;
    }
    const bool ok = current->ok;
    error = current->error;
    if (pcm_out) pcm_out->assign((const char *) current->result.data(), current->result.size());
    g_service.request.reset();
    g_service.cv.notify_all();
    return ok;
}

void cosyvoice3_service_stop() {
    std::thread thread;
    {
        std::lock_guard<std::mutex> lock(g_service.mu);
        if (!g_service.thread.joinable()) return;
        g_service.stopping = true;
        g_service.cv.notify_all();
        thread = std::move(g_service.thread);
    }
    if (thread.joinable()) thread.join();
    std::lock_guard<std::mutex> lock(g_service.mu);
    g_service.ready = false;
    g_service.failed = false;
    g_service.starting = false;
    g_service.stopping = false;
    g_service.request.reset();
}

bool cosyvoice3_service_available() {
    std::lock_guard<std::mutex> lock(g_service.mu);
    return g_service.ready && !g_service.failed && !g_service.stopping;
}
