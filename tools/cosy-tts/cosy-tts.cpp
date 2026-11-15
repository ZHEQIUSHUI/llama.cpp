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

#include "onnxruntime_cxx_api.h"
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
    size_t n = 0;
    if (fmt == 1 && bits == 16) n = data.size() / 2;
    else if (fmt == 3 && bits == 32) n = data.size() / 4;
    else { fprintf(stderr, "%s: unsupported wav fmt=%u bits=%u\n", path, fmt, bits); return false; }
    out.pcm.resize(n / channels);
    for (size_t i = 0; i < out.pcm.size(); i++) {
        float acc = 0;
        for (uint16_t c = 0; c < channels; c++) {
            size_t j = i * channels + c;
            acc += (fmt == 1) ? ((const int16_t *) data.data())[j] / 32768.0f
                              : ((const float *) data.data())[j];
        }
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
    const double limit = std::max(8.0, gen_cap * 0.95);
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
// ONNX Runtime helpers

struct ort_session {
    Ort::Session session{nullptr};
    std::vector<std::string> in_names, out_names;
    std::vector<const char *> in_c, out_c;

    ort_session() = default;
    ort_session(Ort::Env & env, const std::string & path,
                GraphOptimizationLevel level = GraphOptimizationLevel::ORT_ENABLE_ALL) {
        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(level);
        opt.SetIntraOpNumThreads(16);
        session = Ort::Session(env, path.c_str(), opt);
        for (size_t i = 0; i < session.GetInputCount(); i++) {
            auto n = session.GetInputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            in_names.emplace_back(n.get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); i++) {
            auto n = session.GetOutputNameAllocated(i, Ort::AllocatorWithDefaultOptions());
            out_names.emplace_back(n.get());
        }
        for (auto & n : in_names) in_c.push_back(n.c_str());
        for (auto & n : out_names) out_c.push_back(n.c_str());
    }

    std::vector<Ort::Value> run(std::vector<Ort::Value> inputs) {
        return session.Run(Ort::RunOptions{nullptr}, in_c.data(), inputs.data(), inputs.size(), out_c.data(), out_c.size());
    }
};

struct ort_alloc {
    Ort::MemoryInfo info{nullptr};
    ort_alloc() : info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {}
    template <typename T>
    Ort::Value make(std::vector<T> & data, const std::vector<int64_t> & shape) {
        return Ort::Value::CreateTensor<T>(info, data.data(), data.size(), shape.data(), shape.size());
    }
};

// ---------------------------------------------------------------------------
// Load a compiled AXModel whose weights and IO layout are fixed at build time.

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


    const float * output_f32(size_t expected) const {
        if (info->nOutputSize != 1 || info->pOutputs[0].eDataType != AX_ENGINE_DT_FLOAT32 ||
            out[0].nSize < expected * sizeof(float)) {
            fprintf(stderr, "NPU output mismatch: outputs=%u dtype=%d bytes=%u expected=%zu\n",
                    info->nOutputSize, (int) info->pOutputs[0].eDataType, out[0].nSize, expected * sizeof(float));
            return nullptr;
        }
        return (const float *) out[0].pVirAddr;
    }

    size_t output_bytes() const {
        return info->nOutputSize == 1 ? info->pOutputs[0].nSize : 0;
    }

    ~npu_model() {
        for (auto & b : in)  if (b.pVirAddr) AX_SYS_MemFree(b.phyAddr, b.pVirAddr);
        for (auto & b : out) if (b.pVirAddr) AX_SYS_MemFree(b.phyAddr, b.pVirAddr);
        if (handle) AX_ENGINE_DestroyHandle(handle);
    }
};
#else
struct npu_model {
    bool load(const std::string &) {
        fprintf(stderr, "this build has no AXERA support; rebuild with GGML_AXERA and GGML_AXERA_BSP_DIR\n");
        return false;
    }
    bool set_input(const char *, const void *, size_t) { return false; }
    bool run() { return false; }
    const float * output_f32(size_t) const { return nullptr; }
    size_t output_bytes() const { return 0; }
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
    const uint8_t * map = nullptr;
    size_t map_size = 0;

    ~hift_native() {
        if (map) munmap((void *) map, map_size);
    }

    bool load(const std::string & path) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) return false;
        struct stat st{};
        if (fstat(fd, &st) != 0 || st.st_size < 64) { close(fd); return false; }
        map_size = (size_t) st.st_size;
        map = (const uint8_t *) mmap(nullptr, map_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (map == MAP_FAILED) { map = nullptr; return false; }
        const uint8_t * p = map;
        const uint8_t * end = map + map_size;
        auto take = [&](size_t n) { const float * r = (const float *) p; p += n; return r; };
        if (memcmp(p, "CHFTNAT2", 8) != 0) { fprintf(stderr, "bad native HiFT table\n"); return false; }
        p += 8;
        uint32_t n = *(const uint32_t *) p; p += 4;
        layers.clear();
        for (uint32_t i = 0; i < n; i++) {
            if (p + 12 > end) return false;
            layer l;
            l.out_c = (int) *(const uint32_t *) (p + 0);
            l.in_c  = (int) *(const uint32_t *) (p + 4);
            l.k     = (int) *(const uint32_t *) (p + 8);
            p += 12;
            const size_t wb = (size_t) l.out_c * l.in_c * l.k;
            l.w = take(wb * 4);
            l.b = take((size_t) l.out_c * 4);
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
        classifier_w = take(512 * 4);
        classifier_b = take(4);
        linear_w = take(9 * 4);
        linear_b = take(4);
        rand_ini = take(9 * 4);
        if (p + 8 > end) return false;
        noise_samples = *(const uint64_t *) p; p += 8;
        noise = take(noise_samples * 9 * 4);
        if (p > end) return false;
        fprintf(stderr, "[hift-native] loaded %u layers, noise_samples=%zu\n", n, noise_samples);
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
                              size_t sample_offset = 0, double * phase_frames = nullptr) {
    constexpr int HARM = 9, UPS = 480, SR = 24000, HOP = 4, KB = 9;
    const auto clock_now = [] { return std::chrono::steady_clock::now(); };
    const auto elapsed = [](auto a, auto b) { return std::chrono::duration<double>(b - a).count(); };
    const auto t0 = clock_now();
    std::vector<float> a((size_t) T * 80), b;
    for (int c = 0; c < 80; c++) {
        for (int t = 0; t < T; t++) a[(size_t) t * 80 + c] = mel[(size_t) c * T + t];
    }
    for (int li = 0; li < (int) h.layers.size(); li++) {
        conv_causal_parallel(a.data(), T, h.layers[li], li == 0, b);
        a.swap(b);
    }
    const auto t1 = clock_now();
    const int C = h.layers.back().out_c;
    std::vector<float> f0(T);
    for (int t = 0; t < T; t++) {
        double sum = h.classifier_b[0];
        for (int c = 0; c < C; c++) sum += (double) a[(size_t) t * C + c] * h.classifier_w[c];
        f0[t] = std::fabs((float) sum);
    }

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

struct feature_tables {
    std::vector<float> whisper_mel;    // 128 x 201
    std::vector<float> kaldi_mel;      // 80 x 257
    std::vector<float> whisper_hann;   // 400
    std::vector<float> kaldi_povey;    // 512
    std::vector<float> flow_noise;     // 1 x 80 x 15000
    std::vector<float> flow_mel;       // 80 x 961
    std::vector<float> flow_hann;      // 1920
};

static feature_tables load_tables(const std::string & dir) {
    feature_tables t;
    t.whisper_mel  = load_f32(dir + "/whisper_mel_128x201.f32", 128 * 201);
    t.kaldi_mel    = load_f32(dir + "/kaldi_mel_80x257.f32", 80 * 257);
    t.whisper_hann = load_f32(dir + "/whisper_hann400.f32", 400);
    t.kaldi_povey  = load_f32(dir + "/kaldi_povey512.f32", 512);
    t.flow_noise   = load_f32(dir + "/flow_noise_1x80x15000.f32", 80 * 15000);
    t.flow_mel     = load_f32(dir + "/flow_mel_80x961.f32", 80 * 961);
    t.flow_hann    = load_f32(dir + "/flow_hann1920.f32", 1920);
    return t;
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
    cache.filters.n_mel = n_bins;
    cache.filters.n_fft = 1920;
    cache.filters.data.assign((size_t) n_bins * n_bins, 0.0f);
    for (int i = 0; i < n_bins; i++) cache.filters.data[(size_t) i * n_bins + i] = 1.0f;

    const int n_fft = 1920, hop = 480, pad = (n_fft - hop) / 2;
    std::vector<float> x(pcm24.size() + 2 * pad);
    for (int i = 0; i < pad; i++) x[i] = pcm24[pad - i];
    std::copy(pcm24.begin(), pcm24.end(), x.begin() + pad);
    for (int i = 0; i < pad; i++) x[pcm24.size() + pad + i] = pcm24[pcm24.size() - 2 - i];

    n_frames = (int) (pcm24.size() / hop);
    std::vector<float> feat((size_t) n_frames * 80);
    std::vector<float> frame(n_fft), log_power(n_bins), magnitude(n_bins);
    const float floor_log = std::log(1e-5f);
    for (int i = 0; i < n_frames; i++) {
        std::copy(x.begin() + (size_t) i * hop, x.begin() + (size_t) i * hop + n_fft, frame.begin());
        float * dst = feat.data() + (size_t) i * 80;
        mtmd_audio_log_mel_frame(cache, frame.data(), n_fft, n_bins, log_power.data());
        for (int k = 0; k < n_bins; k++) {
            const float power = std::exp(2.302585092994046f * log_power[k]);
            magnitude[k] = std::sqrt(power + 1e-9f);
        }
        for (int m = 0; m < 80; m++) {
            double sum = 0.0;
            const float * filter = t.flow_mel.data() + (size_t) m * n_bins;
            for (int k = 0; k < n_bins; k++) sum += (double) filter[k] * magnitude[k];
            dst[m] = std::max(std::log((float) sum), floor_log);
        }
    }
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
    std::string model, aux, tables;
    std::string campplus, tokenizer, flow_encoder, flow_estimator, hift;
    std::string flow_npu, flow_npu_long, flow_loop_npu, hift_npu, hift_front, hift_post;
    std::string hift_npu_long, hift_front_long, hift_post_long;
    bool flow_npu_nlc = false, flow_npu_long_nlc = false;
    int flow_npu_cap = 512, flow_npu_long_cap = 1024, hift_npu_long_cap = 1024;
    int flow_prompt_min = 32;
    int hift_left = 64;
    std::string text, prompt_text, prompt_wav, out = "out.wav";
    int ngl = 0, ctx = 4096, threads = 16;
    uint32_t seed = 1986;
    bool greedy = false;
    bool dump_codes = false, mel_only = false;
    bool worker = false;
    int repeat = 1;
    std::string parity_in, parity_out, dump_feats, codes_in, dump_mel, dump_hift_s, dump_hift_mp;
    std::string feat_cache;
    std::string warm_prompt, warm_text;
};

static bool parse_args(int argc, char ** argv, args & a) {
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto val = [&]() -> std::string { if (++i >= argc) { fprintf(stderr, "missing value for %s\n", k.c_str()); std::exit(1); } return argv[i]; };
        if (k == "-m" || k == "--model") a.model = val();
        else if (k == "--aux") a.aux = val();
        else if (k == "--tables") a.tables = val();
        else if (k == "--campplus") a.campplus = val();
        else if (k == "--tokenizer") a.tokenizer = val();
        else if (k == "--flow-encoder") a.flow_encoder = val();
        else if (k == "--flow-estimator") a.flow_estimator = val();
        else if (k == "--hift") a.hift = val();
        else if (k == "--flow-npu") a.flow_npu = val();
        else if (k == "--flow-loop-npu") a.flow_loop_npu = val();
        else if (k == "--flow-prompt-min") a.flow_prompt_min = std::max(0, std::stoi(val()));
        else if (k == "--hift-left") a.hift_left = std::max(8, std::stoi(val()));
        else if (k == "--flow-npu-long") a.flow_npu_long = val();
        else if (k == "--hift-npu") a.hift_npu = val();
        else if (k == "--hift-front") a.hift_front = val();
        else if (k == "--hift-post") a.hift_post = val();
        else if (k == "--hift-npu-long") a.hift_npu_long = val();
        else if (k == "--hift-front-long") a.hift_front_long = val();
        else if (k == "--hift-post-long") a.hift_post_long = val();
        else if (k == "--hift-npu-long-cap") a.hift_npu_long_cap = std::stoi(val());
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
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && !a.aux.empty() && !a.tables.empty() && !a.text.empty() && !a.prompt_wav.empty() &&
           !a.campplus.empty() && !a.tokenizer.empty() &&
           (!a.flow_loop_npu.empty() || (!a.flow_encoder.empty() &&
            (!a.flow_npu.empty() || !a.flow_estimator.empty()))) &&
           (!a.hift_npu.empty() || !a.hift.empty());
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

template <typename Fn>
static void parallel_ranges(int n, int max_threads, Fn && fn) {
    const int nt = std::max(1, std::min(max_threads, n));
    if (nt == 1) {
        fn(0, n);
        return;
    }
    std::vector<std::thread> workers;
    workers.reserve(nt - 1);
    for (int i = 1; i < nt; i++) {
        const int a = (n * i) / nt, b = (n * (i + 1)) / nt;
        workers.emplace_back([=, &fn] { fn(a, b); });
    }
    fn(0, n / nt);
    for (auto & worker : workers) worker.join();
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
            "usage: %s -m llm.gguf --aux aux.gguf --tables DIR --campplus campplus.onnx\n"
            "  --tokenizer speech_tokenizer_v3.onnx --flow-encoder flow_encoder.onnx\n"
            "  --flow-estimator flow.decoder.estimator.fp32.onnx --hift hift.onnx\n"
            "  -p TEXT --prompt-text TEXT --prompt-wav ref.wav -o out.wav [-ngl N] [--dump-codes]\n"
            "  (or --flow-npu flow.axmodel [--flow-npu-nlc] --hift-npu hift.axmodel instead of the ONNX stages)\n", argv[0]);
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

    feature_tables tab = load_tables(a.tables);
    hift_native native_hift;
    const bool have_native_hift = native_hift.load(a.tables + "/hift_front.bin");
    fprintf(stderr, "[flow-cfg] prompt_min=%d hift_left=%d\n",
            a.flow_prompt_min, a.hift_left);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cosyvoice3");
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto make_session = [&](const std::string & path,
                            GraphOptimizationLevel level = GraphOptimizationLevel::ORT_ENABLE_ALL) {
        return std::make_unique<ort_session>(env, path, level);
    };
    double t_sess = 0.0;
    auto session = [&](std::unique_ptr<ort_session> & slot, const std::string & path,
                       GraphOptimizationLevel level = GraphOptimizationLevel::ORT_ENABLE_ALL) -> ort_session & {
        if (!slot) {
            auto s0 = std::chrono::steady_clock::now();
            slot = make_session(path, level);
            t_sess += sec(s0, std::chrono::steady_clock::now());
        }
        return *slot;
    };
    // ORT >= 1.23 fuses this graph in a way that changes the FP32 result by
    // ~1%; the reference path keeps its unoptimized computation.
    std::unique_ptr<ort_session> campplus, tokenizer, flow_enc, estimator, hift, hift_front, hift_post;
    std::unique_ptr<npu_model> flow_npu, flow_npu_long, flow_loop_npu, hift_npu, hift_npu_long;
    double t_npu_load = 0.0;
    if (!a.flow_loop_npu.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        flow_loop_npu = std::make_unique<npu_model>();
        if (!flow_loop_npu->load(a.flow_loop_npu)) return 1;
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    }
    if (!a.flow_npu.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        flow_npu = std::make_unique<npu_model>();
        if (!flow_npu->load(a.flow_npu)) return 1;
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    } else if (a.flow_loop_npu.empty()) {
        session(estimator, a.flow_estimator);
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
    if (!a.hift_npu_long.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        hift_npu_long = std::make_unique<npu_model>();
        if (!hift_npu_long->load(a.hift_npu_long)) return 1;
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
        auto pcm16 = resample_sinc(wav.pcm, wav.sample_rate, 16000);
        auto pcm24 = resample_sinc(wav.pcm, wav.sample_rate, 24000);
        const uint64_t feat_key = feat_hash(pcm16.data(), pcm16.size() * sizeof(float)) ^
                                  ((uint64_t) pcm16.size() * 0x9e3779b97f4a7c15ull);
        auto hit = ref_mem.find(feat_key);
        if (hit != ref_mem.end()) {
            fprintf(stderr, "[frontend] cache=memory total=%.3fs speech_tokens=%zu\n",
                    sec(ts0, now()), hit->second->tok.size());
            return hit->second;
        }
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
            rf->fbank = kaldi_fbank80(pcm16, rf->fbank_frames, tab);
            auto t1 = now();
            std::vector<int64_t> spk_shape{1, rf->fbank_frames, 80};
            Ort::Value spk_in = Ort::Value::CreateTensor<float>(mem, rf->fbank.data(), rf->fbank.size(), spk_shape.data(), spk_shape.size());
            std::vector<Ort::Value> spk_inputs;
            spk_inputs.emplace_back(std::move(spk_in));
            auto spk_out = session(campplus, a.campplus, GraphOptimizationLevel::ORT_DISABLE_ALL).run(std::move(spk_inputs));
            const float * spk_data = spk_out[0].GetTensorData<float>();
            rf->embedding.assign(spk_data, spk_data + 192);
            auto t2 = now();
            rf->wmel = whisper_mel(pcm16, rf->wmel_frames, tab);
            auto t3 = now();
            std::vector<int64_t> wmel_shape{1, 128, rf->wmel_frames};
            std::vector<int32_t> wmel_len{rf->wmel_frames};
            std::vector<int64_t> one_shape{1};
            std::vector<Ort::Value> tok_in;
            tok_in.emplace_back(Ort::Value::CreateTensor<float>(mem, rf->wmel.data(), rf->wmel.size(), wmel_shape.data(), wmel_shape.size()));
            tok_in.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, wmel_len.data(), 1, one_shape.data(), one_shape.size()));
            auto tok_out = session(tokenizer, a.tokenizer).run(std::move(tok_in));
            const int32_t * tok_data = tok_out[0].GetTensorData<int32_t>();
            auto tok_info = tok_out[0].GetTensorTypeAndShapeInfo();
            size_t n_prompt_tok = 1;
            for (auto d : tok_info.GetShape()) n_prompt_tok *= (size_t) d;
            rf->tok.assign(tok_data, tok_data + n_prompt_tok);
            auto t4 = now();
            rf->flowfeat = flow_mel24(pcm24, rf->flow_frames, tab);
            auto t5 = now();
            t_fbank   = sec(t0, t1);
            t_spk     = sec(t1, t2);
            t_wmel    = sec(t2, t3);
            t_stok    = sec(t3, t4);
            t_flowmel = sec(t4, t5);
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
        ref_mem.emplace(feat_key, rf);
        return rf;
    };

    // ---- Qwen2 backbone -------------------------------------------------
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
    // The reference prompt (sos + prompt text + task id + prompt speech
    // tokens) is identical for repeated requests with the same reference and
    // text.  Keep its KV in sequence 1 and copy it back instead of re-prefilling
    // ~100 rows every request.  The AXERA backend stores the page images in
    // core/seqstate, so seq_cp moves the private pages too.
    struct prompt_kv_cache {
        bool valid = false;
        std::vector<int32_t> ids;
        int rows = 0;
        std::vector<float> logits;
    };
    prompt_kv_cache pcache;
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
    const auto t_kv_hit0 = std::chrono::steady_clock::now();
    const bool prefix_valid = pcache.valid && pcache.rows == n_prompt_rows;
    const bool prefix_ids_match = prefix_valid && pcache.ids == prefix_key;
    if (codes_path.empty() && pcache.valid && pcache.rows == n_prompt_rows &&
        pcache.ids == prefix_key) {
        llama_memory_seq_cp(kv_mem, 1, 0, -1, -1);
        prefix_hit = true;
    } else {
        // seq 1 holds the previous prompt's pages.  They must be released
        // before a new prompt can receive consecutive cells from cell 0; a
        // seq_rm(seq 0) alone leaves those pages owned by seq 1 and shifts the
        // new prefill to high physical cells, which breaks seqstate's page
        // boundaries.
        llama_memory_seq_rm(kv_mem, 1, -1, -1);
    }
    if (codes_path.empty() && getenv("COSY_LLM_PROBE")) {
        fprintf(stderr, "[llm-cache] valid=%d rows=%d/%d ids_match=%d key=%zu\n",
                pcache.valid ? 1 : 0, pcache.rows, n_prompt_rows, prefix_ids_match ? 1 : 0,
                prefix_key.size());
    }
    const double t_kv_hit = sec(t_kv_hit0, std::chrono::steady_clock::now());

    std::vector<float> h;
    llama_batch_guard batch(512, n_embd);
    const bool streaming = (bool) stream_cb && incremental;
    const int stream_tok_cap = flow_loop_npu ? (int) flow_loop_npu->input_elems("tokens") : 0;
    const int stream_mel_cap = flow_loop_npu ? (int) (flow_loop_npu->input_elems("noise") / 80) : 0;
    const int stream_prompt_n = std::min((int) prompt_tok.size(), std::max(0, stream_tok_cap - 16));
    const int stream_prompt_mel = 2 * stream_prompt_n;
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
            stream_error = "streaming requires merged flow and native HiFT";
            return false;
        }
        // Official streaming semantics: rerun the flow window from position
        // zero on the token prefix generated so far, then emit only the mel
        // frames past the previous offset.  The fixed window is never
        // continued from an independent left-context window, so the result is
        // exactly the non-stream output for everything emitted so far.
        if (stream_prompt_n + available > stream_tok_cap) {
            stream_error = "streaming segment exceeds the flow window";
            return false;
        }
        std::vector<int32_t> tokens(stream_tok_cap, 0);
        for (int i = 0; i < stream_prompt_n; i++) tokens[i] = prompt_tok[i];
        for (int i = 0; i < available; i++) tokens[stream_prompt_n + i] = stream_flow_codes[i];
        const int32_t gen_len_i = available;
        const int32_t prompt_len_i = stream_prompt_n;
        const int valid_tokens = stream_prompt_n + available;
        const bool mask_inputs = flow_loop_npu->input_elems("token_mask") > 0;
        std::vector<float> token_mask(mask_inputs ? stream_tok_cap : 0, 0.0f);
        std::vector<float> mel_mask(mask_inputs ? stream_mel_cap : 0, 0.0f);
        if (mask_inputs) {
            std::fill_n(token_mask.begin(), valid_tokens, 1.0f);
            std::fill_n(mel_mask.begin(), 2 * valid_tokens, 1.0f);
        }

        std::vector<float> pfeat((size_t) stream_mel_cap * 80, 0.0f);
        const int pframes = std::min({(int) rf.flow_frames, stream_prompt_mel, stream_mel_cap});
        memcpy(pfeat.data(), rf.flowfeat.data(), (size_t) pframes * 80 * sizeof(float));
        std::vector<float> noise((size_t) 80 * stream_mel_cap, 0.0f);
        for (int c = 0; c < 80; c++) {
            memcpy(noise.data() + (size_t) c * stream_mel_cap,
                   tab.flow_noise.data() + (size_t) c * 15000, (size_t) stream_mel_cap * sizeof(float));
        }
        std::vector<float> emb = rf.embedding;
        auto tf0 = std::chrono::steady_clock::now();
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
                  flow_loop_npu->run();
        if (!flow_ok) {
            stream_error = "streaming flow kernel failed";
            return false;
        }
        const float * out = flow_loop_npu->output_f32((size_t) 80 * stream_mel_cap);
        if (!out) {
            stream_error = "streaming flow produced no output";
            return false;
        }
        stream_flow_s += sec(tf0, std::chrono::steady_clock::now());
        const int out_start = stream_prompt_mel + 2 * stream_out_codes;
        const int new_frames = 2 * (available - stream_out_codes);
        const int old_frames = (int) stream_mel_ch[0].size();
        const int history_start = std::max(0, old_frames - a.hift_left);
        const int history = old_frames - history_start;
        const int dst_off = a.hift_left - history;

        std::vector<float> front_mel((size_t) 80 * (history + new_frames));
        for (int c = 0; c < 80; c++) {
            if (history) memcpy(front_mel.data() + (size_t) c * (history + new_frames),
                                stream_mel_ch[c].data() + history_start, (size_t) history * 4);
            memcpy(front_mel.data() + (size_t) c * (history + new_frames) + history,
                   out + (size_t) c * stream_mel_cap + out_start, (size_t) new_frames * 4);
        }
        std::vector<float> s_front;
        int s_front_frames = 0;
        std::vector<double> s_phase((size_t) (history + new_frames) * 9);
        const double * phase0 = history_start ? stream_phase_frames.data() + (size_t) (history_start - 1) * 9
                                              : nullptr;
        const auto tfront0 = std::chrono::steady_clock::now();
        hift_front_native(native_hift, front_mel.data(), history + new_frames, s_front, s_front_frames,
                          phase0, nullptr, (size_t) history_start * 480, s_phase.data());
        stream_front_s += sec(tfront0, std::chrono::steady_clock::now());
        stream_phase_frames.resize((size_t) (old_frames + new_frames) * 9);
        memcpy(stream_phase_frames.data() + (size_t) old_frames * 9,
               s_phase.data() + (size_t) history * 9, (size_t) new_frames * 9 * sizeof(double));

        const int cap = (int) (hift_npu->input_elems("mel") / 80);
        if (cap <= 64) {
            stream_error = "streaming HiFT kernel has an invalid window";
            return false;
        }
        const int s_frames = cap * 120 + 1;
        std::vector<float> hmel((size_t) 80 * cap, 0.0f);
        std::vector<float> s_chunk((size_t) 18 * s_frames, 0.0f);
        for (int c = 0; c < 80; c++) {
            if (history) memcpy(hmel.data() + (size_t) c * cap + dst_off,
                                stream_mel_ch[c].data() + history_start, (size_t) history * 4);
            memcpy(hmel.data() + (size_t) c * cap + 64,
                   out + (size_t) c * stream_mel_cap + out_start, (size_t) new_frames * 4);
        }
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
        const size_t mp_bytes = hift_npu->output_bytes();
        const int mp_ch = mp_bytes ? (int) (mp_bytes / (sizeof(float) * (size_t) s_frames)) : 0;
        const float * mp = hift_npu->output_f32((size_t) mp_ch * s_frames);
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
        for (int c = 0; c < 80; c++) {
            stream_mel_ch[c].insert(stream_mel_ch[c].end(),
                                    out + (size_t) c * stream_mel_cap + out_start,
                                    out + (size_t) c * stream_mel_cap + out_start + new_frames);
        }
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
            logits = pcache.logits;
            fprintf(stderr, "[llm] prefix cache hit rows=%d\n", n_prompt_rows);
        } else {
            const auto t_l0 = std::chrono::steady_clock::now();
            if (!read_logits(logits)) {
                fprintf(stderr, "no speech logits from the prefill graph\n");
                return 1;
            }
            t_logits_initial = sec(t_l0, std::chrono::steady_clock::now());
            const auto t_c0 = std::chrono::steady_clock::now();
            llama_memory_seq_rm(kv_mem, 1, -1, -1);
            llama_memory_seq_cp(kv_mem, 0, 1, 0, n_prompt_rows);
            t_cache_cp = sec(t_c0, std::chrono::steady_clock::now());
            pcache.valid = true;
            pcache.rows = n_prompt_rows;
            pcache.ids = std::move(prefix_key);
            pcache.logits = logits;
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
            // One flow call covers the complete token prefix.  Leave room for
            // the minimum reference prefix; the flow window itself trims the
            // reference as needed instead of forcing an extra full call.
            const int tok_cap = (int) flow_loop_npu->input_elems("tokens");
            const int prompt_floor = std::min((int) prompt_tok.size(),
                                              std::max(8, a.flow_prompt_min));
            max_len = std::min(max_len, std::max(1, tok_cap - prompt_floor - 4));
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
                const int threshold = stream_out_codes == 0 ? 4 : 64;
                if ((int) stream_flow_codes.size() - stream_out_codes >= threshold && !run_stream_window()) {
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
    auto shape_of = [](const Ort::Value & v) { return v.GetTensorTypeAndShapeInfo().GetShape(); };
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
        // Keep the longest reference prefix that leaves room for this
        // sentence, with --flow-prompt-min as the lower bound.  The speaker
        // embedding and the language-model prompt remain complete.
        const int prompt_floor = std::min(full_prompt_n, std::max(8, a.flow_prompt_min));
        const int prompt_n = std::min(full_prompt_n,
                                      std::max(prompt_floor, tok_cap - (int) flow_codes.size() - 4));
        const int prompt_mel = 2 * prompt_n;
        if (full_prompt_n > prompt_n) {
            fprintf(stderr, "[flow] reference prompt %d tokens truncated to %d for the merged window\n",
                    full_prompt_n, prompt_n);
        }
        Tgen = 2 * (int) flow_codes.size();
        T_total = prompt_mel + Tgen;
        mel.assign((size_t) 80 * Tgen, 0.0f);
        std::vector<int32_t> tokens(tok_cap, 0);
        std::vector<float> pfeat((size_t) mel_cap * 80, 0.0f);
        const int pframes = std::min({(int) rf.flow_frames, prompt_mel, mel_cap});
        memcpy(pfeat.data(), rf.flowfeat.data(), (size_t) pframes * 80 * sizeof(float));
        std::vector<float> noise((size_t) 80 * mel_cap, 0.0f);
        std::vector<float> emb = rf.embedding;
        if ((int) emb.size() != 192) { fprintf(stderr, "flow embedding has %zu values\n", emb.size()); return 1; }
        const int total_tokens = prompt_n + (int) flow_codes.size();
        if (total_tokens > tok_cap) {
            fprintf(stderr, "[flow] segment needs %d tokens, window holds %d; split the text first\n",
                    total_tokens, tok_cap);
            return 1;
        }
        std::fill(tokens.begin(), tokens.end(), 0);
        for (int i = 0; i < prompt_n; i++) tokens[i] = prompt_tok[i];
        for (size_t i = 0; i < flow_codes.size(); i++) tokens[prompt_n + i] = flow_codes[i];
        const bool mask_inputs = flow_loop_npu->input_elems("token_mask") > 0;
        std::vector<float> token_mask(mask_inputs ? tok_cap : 0, 0.0f);
        std::vector<float> mel_mask(mask_inputs ? mel_cap : 0, 0.0f);
        if (mask_inputs) {
            std::fill_n(token_mask.begin(), total_tokens, 1.0f);
            std::fill_n(mel_mask.begin(), 2 * total_tokens, 1.0f);
        }
        const int32_t gen_len_i = (int32_t) flow_codes.size();
        const int32_t prompt_len_i = prompt_n;
        for (int c = 0; c < 80; c++) {
            const float * src = tab.flow_noise.data() + (size_t) c * noise_cap;
            memcpy(noise.data() + (size_t) c * mel_cap, src, (size_t) mel_cap * sizeof(float));
        }
        fprintf(stderr, "[flow-win] pn=%d gen=%zu total=%d cap=%d\n",
                prompt_n, flow_codes.size(), total_tokens, tok_cap);
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
                  flow_loop_npu->run();
        if (!flow_ok) return 1;
        const float * out = flow_loop_npu->output_f32((size_t) 80 * mel_cap);
        if (!out) return 1;
        for (int c = 0; c < 80; c++) {
            memcpy(mel.data() + (size_t) c * Tgen,
                   out + (size_t) c * mel_cap + prompt_mel, (size_t) Tgen * sizeof(float));
        }
        fprintf(stderr, "[flow] windows=1 cap_tokens=%d cap_mel=%d\n", tok_cap, mel_cap);
        t_flow = std::chrono::steady_clock::now();
    } else {
    std::vector<int32_t> gen_tok(flow_codes.begin(), flow_codes.end());
    std::vector<int64_t> tok_shape{1, (int64_t) gen_tok.size()};
    std::vector<int64_t> ptok_shape{1, (int64_t) prompt_tok.size()};
    std::vector<int64_t> pfeat_shape{1, rf.flow_frames, 80};
    std::vector<int64_t> emb_shape{1, 192};
    std::vector<Ort::Value> fe_in;
    fe_in.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, gen_tok.data(), gen_tok.size(), tok_shape.data(), tok_shape.size()));
    fe_in.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, const_cast<int32_t *>(prompt_tok.data()), prompt_tok.size(), ptok_shape.data(), ptok_shape.size()));
    fe_in.emplace_back(Ort::Value::CreateTensor<float>(mem, const_cast<float *>(prompt_feat.data()), prompt_feat.size(), pfeat_shape.data(), pfeat_shape.size()));
    fe_in.emplace_back(Ort::Value::CreateTensor<float>(mem, const_cast<float *>(embedding.data()), embedding.size(), emb_shape.data(), emb_shape.size()));
    auto fe_out = session(flow_enc, a.flow_encoder).run(std::move(fe_in));
    auto mu_shape = shape_of(fe_out[0]);
    auto cond_shape = shape_of(fe_out[1]);
    auto spks_shape = shape_of(fe_out[2]);
    auto mask_shape = shape_of(fe_out[3]);
    if (mu_shape.size() != 3) { fprintf(stderr, "bad flow mu shape\n"); return 1; }
    const int T = (int) mu_shape[2];
    T_total = T;
    if (T != 2 * ((int) prompt_tok.size() + (int) flow_codes.size())) {
        fprintf(stderr, "flow length mismatch T=%d prompt_tok=%zu codes=%zu\n", T, prompt_tok.size(), flow_codes.size());
        return 1;
    }
    std::vector<float> mu(T * 80), cond(T * 80), spks(80), mask(T);
    memcpy(mu.data(), fe_out[0].GetTensorData<float>(), mu.size() * 4);
    memcpy(cond.data(), fe_out[1].GetTensorData<float>(), cond.size() * 4);
    memcpy(spks.data(), fe_out[2].GetTensorData<float>(), 80 * 4);
    memcpy(mask.data(), fe_out[3].GetTensorData<float>(), T * 4);
    t_flowenc = std::chrono::steady_clock::now();

    // ---- Euler flow matching -------------------------------------------
    const int steps = 10;
    npu_model * flow_model = flow_npu.get();
    int npu_flow_cap = a.flow_npu_cap;
    bool npu_flow_nlc = a.flow_npu_nlc;
    if (T > npu_flow_cap && flow_npu_long) {
        flow_model = flow_npu_long.get();
        npu_flow_cap = a.flow_npu_long_cap;
        npu_flow_nlc = a.flow_npu_long_nlc;
    }
    std::vector<double> ts(steps + 1);
    for (int i = 0; i <= steps; i++) ts[i] = 1.0 - std::cos(0.5 * M_PI * (double) i / steps);
    std::vector<float> x((size_t) 80 * T);
    for (int c = 0; c < 80; c++)
        for (int t = 0; t < T; t++)
            x[(size_t) c * T + t] = tab.flow_noise[(size_t) c * 15000 + t];

    // One Euler trajectory over Tw frames (Tw <= the kernel's capacity when it runs on NPU).
    auto euler = [&](std::vector<float> & xw, const float * muw, const float * condw,
                     const float * maskw, int Tw) -> bool {
        const int cap = flow_model ? npu_flow_cap : Tw;
        std::vector<float> x_in(2 * 80 * Tw), mu_in(2 * 80 * Tw), cond_in(2 * 80 * Tw);
        std::vector<float> mask_in(2 * Tw), spks_in(2 * 80), t_in(2);
        std::vector<int64_t> x_shape{2, 80, Tw}, mask2_shape{2, 1, Tw}, spks2_shape{2, 80}, t2_shape{2};
        double t_cur = ts[0], dt = ts[1] - ts[0];
        for (int step = 1; step <= steps; step++) {
            const float * d = nullptr;
            std::vector<Ort::Value> es_out;
            if (flow_model) {
                const size_t plane = (size_t) 80 * cap;
                std::vector<float> nx(2 * plane), nmu(2 * plane), ncond(2 * plane);
                std::vector<float> nmask(2 * cap), nspks(2 * 80), nt(2);
                if (npu_flow_nlc) {
                    for (int t = 0; t < Tw; t++) {
                        for (int c = 0; c < 80; c++) {
                            const size_t src = (size_t) c * Tw + t;
                            nx[(size_t) t * 80 + c] = xw[src];
                            nx[plane + (size_t) t * 80 + c] = xw[src];
                            nmu[(size_t) t * 80 + c] = muw[src];
                            ncond[(size_t) t * 80 + c] = condw[src];
                        }
                    }
                } else {
                    for (int c = 0; c < 80; c++) {
                        memcpy(nx.data() + (size_t) c * cap, xw.data() + (size_t) c * Tw, Tw * 4);
                        memcpy(nmu.data() + (size_t) c * cap, muw + (size_t) c * Tw, Tw * 4);
                        memcpy(ncond.data() + (size_t) c * cap, condw + (size_t) c * Tw, Tw * 4);
                        memcpy(nx.data() + plane + (size_t) c * cap, xw.data() + (size_t) c * Tw, Tw * 4);
                    }
                }
                memcpy(nmask.data(), maskw, (size_t) Tw * 4);
                memcpy(nmask.data() + cap, maskw, (size_t) Tw * 4);
                memcpy(nspks.data(), spks.data(), 80 * 4);
                nt[0] = nt[1] = (float) t_cur;
                if (!flow_model->set_input("x", nx.data(), nx.size() * 4) ||
                    !flow_model->set_input("mask", nmask.data(), nmask.size() * 4) ||
                    !flow_model->set_input("mu", nmu.data(), nmu.size() * 4) ||
                    !flow_model->set_input("t", nt.data(), nt.size() * 4) ||
                    !flow_model->set_input("spks", nspks.data(), nspks.size() * 4) ||
                    !flow_model->set_input("cond", ncond.data(), ncond.size() * 4) || !flow_model->run()) {
                    return false;
                }
                d = flow_model->output_f32(2 * plane);
                if (!d) return false;
            } else {
                for (int c = 0; c < 80; c++) {
                    memcpy(x_in.data() + (size_t) c * Tw, xw.data() + (size_t) c * Tw, Tw * 4);
                    memcpy(x_in.data() + (size_t) (80 + c) * Tw, xw.data() + (size_t) c * Tw, Tw * 4);
                    memcpy(mu_in.data() + (size_t) c * Tw, muw + (size_t) c * Tw, Tw * 4);
                    memset(mu_in.data() + (size_t) (80 + c) * Tw, 0, Tw * 4);
                    memcpy(cond_in.data() + (size_t) c * Tw, condw + (size_t) c * Tw, Tw * 4);
                    memset(cond_in.data() + (size_t) (80 + c) * Tw, 0, Tw * 4);
                }
                memcpy(mask_in.data(), maskw, (size_t) Tw * 4);
                memcpy(mask_in.data() + Tw, maskw, (size_t) Tw * 4);
                memcpy(spks_in.data(), spks.data(), 80 * 4);
                memset(spks_in.data() + 80, 0, 80 * 4);
                t_in[0] = t_in[1] = (float) t_cur;
                std::vector<Ort::Value> es_in;
                es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, x_in.data(), x_in.size(), x_shape.data(), x_shape.size()));
                es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, mask_in.data(), mask_in.size(), mask2_shape.data(), mask2_shape.size()));
                es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, mu_in.data(), mu_in.size(), x_shape.data(), x_shape.size()));
                es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, t_in.data(), 2, t2_shape.data(), t2_shape.size()));
                es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, spks_in.data(), spks_in.size(), spks2_shape.data(), spks2_shape.size()));
                es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, cond_in.data(), cond_in.size(), x_shape.data(), x_shape.size()));
                es_out = estimator->run(std::move(es_in));
                d = es_out[0].GetTensorData<float>();
            }
            const int dstride = flow_model ? cap : Tw;
            for (int c = 0; c < 80; c++) {
                for (int t = 0; t < Tw; t++) {
                    float dphi = 1.7f * d[(size_t) c * dstride + t] - 0.7f * d[(size_t) (80 + c) * dstride + t];
                    xw[(size_t) c * Tw + t] += (float) dt * dphi;
                }
            }
            t_cur += dt;
            if (step < steps) dt = ts[step + 1] - t_cur;
        }
        return true;
    };

    if (!flow_model) {
        if (!euler(x, mu.data(), cond.data(), mask.data(), T)) return 1;
    } else if (T <= npu_flow_cap) {
        if (!euler(x, mu.data(), cond.data(), mask.data(), T)) return 1;
    } else {
        // The estimator is causal with 50-frame chunks.  Run the one fixed c1024 kernel over windows:
        // a later window keeps 256 frames of left real context, and its output stops at a chunk boundary
        // unless it is the tail of the sequence.  This removes the old fixed-length limit without a second
        // length-specific AXModel.
        const int left = 256;
        int out0 = 0;
        while (out0 < T) {
            const int in0   = std::max(0, out0 - left);
            const int take  = std::min(npu_flow_cap, T - in0);
            const int maxout = in0 + take;
            int out1 = std::min(out0 + (npu_flow_cap - left), maxout);
            if (out1 < T) out1 = (out1 / 50) * 50;
            if (out1 <= out0) out1 = std::min(maxout, out0 + 1);
            std::vector<float> xw((size_t) 80 * take), muw((size_t) 80 * take), condw((size_t) 80 * take),
                                maskw((size_t) take);
            for (int c = 0; c < 80; c++) {
                memcpy(xw.data() + (size_t) c * take, x.data() + (size_t) c * T + in0, (size_t) take * 4);
                memcpy(muw.data() + (size_t) c * take, mu.data() + (size_t) c * T + in0, (size_t) take * 4);
                memcpy(condw.data() + (size_t) c * take, cond.data() + (size_t) c * T + in0, (size_t) take * 4);
            }
            memcpy(maskw.data(), mask.data() + in0, (size_t) take * 4);
            if (!euler(xw, muw.data(), condw.data(), maskw.data(), take)) return 1;
            for (int c = 0; c < 80; c++) {
                memcpy(x.data() + (size_t) c * T + out0, xw.data() + (size_t) c * take + (out0 - in0),
                       (size_t) (out1 - out0) * 4);
            }
            out0 = out1;
        }
    }
    t_flow = std::chrono::steady_clock::now();

    const int prompt_mel = 2 * (int) prompt_tok.size();
    Tgen = T - prompt_mel;
    mel.assign((size_t) 80 * Tgen, 0.0f);
    for (int c = 0; c < 80; c++)
        memcpy(mel.data() + (size_t) c * Tgen, x.data() + (size_t) c * T + prompt_mel, (size_t) Tgen * 4);
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
        const bool split = native || (!a.hift_front.empty() && !a.hift_post.empty());
        if (cap <= 64) {
            fprintf(stderr, "HiFT kernel has an invalid window %d\n", cap);
            return 1;
        }
        const int s_frames = cap * 480 / 4 + 1;
        double t_front = 0.0, t_npu = 0.0, t_post = 0.0;
        int chunks = 0;
        std::vector<float> full_s;
        if (split) {
            auto t0 = std::chrono::steady_clock::now();
            if (native) {
                int nf_native = 0;
                hift_front_native(native_hift, mel.data(), Tgen, full_s, nf_native);
                if (nf_native != Tgen * 120 + 1 || full_s.size() != (size_t) 18 * nf_native) {
                    fprintf(stderr, "native hift front produced a bad shape for %d frames\\n", Tgen);
                    return 1;
                }
            } else {
                std::vector<int64_t> shape{1, 80, Tgen};
                std::vector<Ort::Value> in;
                in.emplace_back(Ort::Value::CreateTensor<float>(mem, mel.data(), mel.size(),
                                                                shape.data(), shape.size()));
                auto out = session(hift_front, a.hift_front).run(std::move(in));
                auto got = shape_of(out[0]);
                size_t n = 1;
                for (auto d : got) n *= (size_t) d;
                if (got.size() != 3 || got[1] != 18 || n != (size_t) 18 * (Tgen * 120 + 1)) {
                    fprintf(stderr, "hift front produced a bad shape for %d frames\\n", Tgen);
                    return 1;
                }
                full_s.assign(out[0].GetTensorData<float>(), out[0].GetTensorData<float>() + n);
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
                } else {
                    std::vector<int64_t> mp_shape{1, 18, s_frames};
                    std::vector<Ort::Value> po_in;
                    po_in.emplace_back(Ort::Value::CreateTensor<float>(mem, const_cast<float *>(mp),
                                                                       (size_t) 18 * s_frames,
                                                                       mp_shape.data(), mp_shape.size()));
                    auto pr = session(hift_post, a.hift_post).run(std::move(po_in));
                    auto pr_shape = shape_of(pr[0]);
                    size_t n_samples = 1;
                    for (auto d : pr_shape) n_samples *= (size_t) d;
                    if (n_samples < (size_t) cap * 480) {
                        fprintf(stderr, "hift post produced %zu samples, expected %d\\n", n_samples, cap * 480);
                        return 1;
                    }
                    decoded.assign(pr[0].GetTensorData<float>(), pr[0].GetTensorData<float>() + (size_t) cap * 480);
                }
                audio = decoded.data();
                auto th3 = std::chrono::steady_clock::now();
                t_npu += sec(th1, th2);
                t_post += sec(th2, th3);
            } else {
                auto th0 = std::chrono::steady_clock::now();
                if (Tgen > cap) {
                    fprintf(stderr, "shift-only HiFT needs the split front end above %d frames\\n", cap);
                    return 1;
                }
                if (!hmodel->set_input("mel", hmel.data(), hmel.size() * 4) || !hmodel->run()) return 1;
                const float * out = hmodel->output_f32((size_t) cap * 480);
                if (!out) return 1;
                decoded.assign(out, out + (size_t) cap * 480);
                audio = decoded.data();
                t_npu += sec(th0, std::chrono::steady_clock::now());
            }
            memcpy(audio_out.data() + (size_t) start * 480, audio + (size_t) off * 480, (size_t) valid * 480 * 4);
            chunks++;
        }
        pcm.swap(audio_out);
        if (pcm.size() != (size_t) Tgen * 480) {
            fprintf(stderr, "hift produced %zu samples, expected %zu\n", pcm.size(), (size_t) Tgen * 480);
            return 1;
        }
        fprintf(stderr, "[hift] chunks=%d cap=%d front=%.3fs npu=%.3fs post=%.3fs\n",
                chunks, cap, t_front, t_npu, t_post);
        hift_calls = chunks;
    } else {
        std::vector<int64_t> hmel_shape{1, 80, Tgen};
        std::vector<Ort::Value> hi_in;
        hi_in.emplace_back(Ort::Value::CreateTensor<float>(mem, mel.data(), mel.size(), hmel_shape.data(), hmel_shape.size()));
        auto hi_out = session(hift, a.hift).run(std::move(hi_in));
        auto audio_shape = shape_of(hi_out[0]);
        const float * audio = hi_out[0].GetTensorData<float>();
        size_t n_samples = 1;
        for (auto d : audio_shape) n_samples *= (size_t) d;
        pcm.assign(audio, audio + n_samples);
        hift_calls = 1;
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
