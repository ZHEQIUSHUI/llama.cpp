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

#include "onnxruntime_cxx_api.h"
#ifdef COSY_TTS_HAVE_AXERA
#include "ax_engine_api.h"
#include "ax_sys_api.h"
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

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

static std::vector<int> filter_flow_silent_tokens(const std::vector<int> & codes) {
    static const int silent[] = {1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323};
    std::vector<int> out;
    out.reserve(codes.size());
    int run = 0;
    for (int tok : codes) {
        bool is_silent = false;
        for (int s : silent) is_silent = is_silent || tok == s;
        if (is_silent) {
            run++;
            if (run > 5) continue;
        } else {
            run = 0;
        }
        out.push_back(tok);
    }
    return out;
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
    std::vector<AX_ENGINE_IO_BUFFER_T> in, out;

    npu_model() = default;
    npu_model(const npu_model &) = delete;
    npu_model & operator=(const npu_model &) = delete;

    bool load(const std::string & path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { fprintf(stderr, "cannot open axmodel %s\n", path.c_str()); return false; }
        std::vector<char> blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (blob.empty() || AX_ENGINE_CreateHandle(&handle, blob.data(), (AX_U32) blob.size()) != 0 ||
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
    explicit sampler(uint32_t seed) : rng(seed) {}

    static std::vector<float> softmax(const std::vector<float> & logits) {
        const float mx = *std::max_element(logits.begin(), logits.end());
        std::vector<float> p(logits.size());
        float sum = 0;
        for (size_t i = 0; i < logits.size(); i++) {
            p[i] = std::isfinite(logits[i]) ? std::exp(logits[i] - mx) : 0.0f;
            sum += p[i];
        }
        if (sum > 0) for (float & v : p) v /= sum;
        return p;
    }

    int multinomial(const std::vector<float> & probs) {
        std::vector<float> cum(probs.size());
        float sum = 0;
        for (size_t i = 0; i < probs.size(); i++) { sum += std::max(0.0f, probs[i]); cum[i] = sum; }
        if (!(sum > 0)) return 0;
        for (float & v : cum) v /= sum;
        cum.back() = 1.0f;
        const double u = rng.uniform();
        int left = 0, right = (int) probs.size();
        while (right - left > 0) {
            const int mid = left + (right - left) / 2;
            if ((double) cum[mid] < u) left = mid + 1; else right = mid;
        }
        return std::min(left, (int) probs.size() - 1);
    }

    int sample(std::vector<float> logits, const std::vector<int> & decoded, bool ignore_eos, int eos) {
        if (ignore_eos && eos >= 0 && eos < (int) logits.size()) logits[eos] = -std::numeric_limits<float>::infinity();
        const std::vector<float> probs = softmax(logits);
        std::vector<int> idx(probs.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });

        std::vector<float> nucleus;
        float cum = 0;
        for (size_t i = 0; i < idx.size(); i++) {
            if (!(cum < 0.8f) || nucleus.size() >= 25) break;
            cum += probs[idx[i]];
            nucleus.push_back(probs[idx[i]]);
        }
        if (nucleus.empty()) nucleus.push_back(probs[idx[0]]);
        int top = idx[multinomial(nucleus)];

        int rep = 0;
        for (int i = std::max(0, (int) decoded.size() - 10); i < (int) decoded.size(); i++) rep += decoded[i] == top;
        if (rep >= 1) {
            logits[top] = -std::numeric_limits<float>::infinity();
            top = multinomial(softmax(logits));
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
    std::string flow_npu, flow_npu_long, hift_npu, hift_front, hift_post;
    std::string hift_npu_long, hift_front_long, hift_post_long;
    bool flow_npu_nlc = false, flow_npu_long_nlc = false;
    int flow_npu_cap = 512, flow_npu_long_cap = 1024, hift_npu_long_cap = 1024;
    std::string text, prompt_text, prompt_wav, out = "out.wav";
    int ngl = 0, ctx = 4096, threads = 16;
    uint32_t seed = 1986;
    bool dump_codes = false, mel_only = false;
    bool worker = false;
    int repeat = 1;
    std::string parity_in, parity_out, dump_feats, codes_in, dump_mel, dump_hift_s, dump_hift_mp;
    std::string feat_cache;
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
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && !a.aux.empty() && !a.tables.empty() && !a.text.empty() && !a.prompt_wav.empty() &&
           !a.campplus.empty() && !a.tokenizer.empty() && !a.flow_encoder.empty() &&
           (!a.flow_npu.empty() || !a.flow_estimator.empty()) && (!a.hift_npu.empty() || !a.hift.empty());
}

static bool axera_engine_init_once() {
#ifdef COSY_TTS_HAVE_AXERA
    static const int rc = [] {
        if (AX_SYS_Init() != 0) return 1;
        AX_ENGINE_NPU_ATTR_T attr{};
        attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
        return AX_ENGINE_Init(&attr) == 0 ? 0 : 2;
    }();
    if (rc) fprintf(stderr, "AXERA engine init failed (%d)\n", rc);
    return rc == 0;
#else
    fprintf(stderr, "this build has no AXERA support; rebuild with GGML_AXERA and GGML_AXERA_BSP_DIR\n");
    return false;
#endif
}

static double sec(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double>(b - a).count();
}

static int decode_embd(llama_context * ctx, const float * embd, int n_tokens, int n_embd, int pos0, bool logits_last) {
    llama_batch batch = llama_batch_init(n_tokens, n_embd, 1);
    batch.n_tokens = n_tokens;
    memcpy(batch.embd, embd, (size_t) n_tokens * n_embd * sizeof(float));
    for (int i = 0; i < n_tokens; i++) {
        batch.pos[i] = pos0 + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = logits_last && i == n_tokens - 1;
    }
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc;
}

int main(int argc, char ** argv) {
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
        if (decode_embd(ctx, in.data(), rows, n_embd, 0, true) != 0) { fprintf(stderr, "parity decode failed\n"); return 1; }
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
    std::unique_ptr<npu_model> flow_npu, flow_npu_long, hift_npu, hift_npu_long;
    double t_npu_load = 0.0;
    if (!a.flow_npu.empty()) {
        if (!axera_engine_init_once()) return 1;
        auto s0 = std::chrono::steady_clock::now();
        flow_npu = std::make_unique<npu_model>();
        if (!flow_npu->load(a.flow_npu)) return 1;
        t_npu_load += sec(s0, std::chrono::steady_clock::now());
    } else {
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
        std::vector<float> wm((size_t) 80 * 512), ws((size_t) 18 * 61441);
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

    auto tokenize = [&](const std::string & text) {
        std::vector<llama_token> v(text.size() + 64);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), v.data(), (int) v.size(), false, true);
        if (n < 0) { v.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int) text.size(), v.data(), (int) v.size(), false, true); }
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); std::exit(1); }
        v.resize(n);
        return v;
    };
    auto generate = [&](const std::string & req_text, const std::string & req_prompt_text,
                        const std::string & outpath, const ref_feats & rf,
                        const std::string & codes_path, double front_s, double llm_load_s) -> int {
    auto t_req0 = std::chrono::steady_clock::now();
    llama_memory_clear(llama_get_memory(ctx), true);
    const std::vector<int32_t> & prompt_tok = rf.tok;
    const std::vector<float> & prompt_feat = rf.flowfeat;
    const std::vector<float> & embedding = rf.embedding;
    auto prompt_text_tok = tokenize(req_prompt_text);
    auto text_tok = tokenize(req_text);
    std::vector<llama_token> text_all = prompt_text_tok;
    text_all.insert(text_all.end(), text_tok.begin(), text_tok.end());

    std::vector<float> text_emb((size_t) text_all.size() * n_embd);
    if (llama_model_get_tok_embd_rows(model, text_all.data(), (uint32_t) text_all.size(), text_emb.data()) == 0) {
        fprintf(stderr, "failed to read token embeddings\n"); return 1;
    }

    auto speech_row = [&](int id) { return aux.speech_emb.data() + (size_t) id * n_embd; };
    std::vector<float> lm_input;
    lm_input.insert(lm_input.end(), speech_row(6561), speech_row(6561) + n_embd); // sos
    lm_input.insert(lm_input.end(), text_emb.begin(), text_emb.end());
    lm_input.insert(lm_input.end(), speech_row(6563), speech_row(6563) + n_embd); // task id
    for (int32_t id : prompt_tok) lm_input.insert(lm_input.end(), speech_row(id), speech_row(id + 1));
    const int n_prompt_rows = (int) (lm_input.size() / n_embd);
    fprintf(stderr, "[llm] prompt_text=%zu text=%zu prompt_speech=%zu rows=%d\n",
            prompt_text_tok.size(), text_tok.size(), prompt_tok.size(), n_prompt_rows);

    std::vector<float> h;
    const float * hidden = nullptr;
    auto t_prefill = std::chrono::steady_clock::now();
    if (codes_path.empty()) {
        if (decode_embd(ctx, lm_input.data(), n_prompt_rows, n_embd, 0, true) != 0) {
            fprintf(stderr, "prefill decode failed\n"); return 1;
        }
        hidden = llama_get_embeddings_ith(ctx, -1);
        if (!hidden) { fprintf(stderr, "no hidden state\n"); return 1; }
        h.assign(hidden, hidden + n_embd);
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
    if (codes_path.empty() && !read_logits(logits)) {
        fprintf(stderr, "no speech logits from the prefill graph\n");
        return 1;
    }
    std::vector<int> codes;
    auto t_gen0 = std::chrono::steady_clock::now();
    if (!codes_path.empty()) {
        if (!load_codes(codes_path, codes)) return 1;
        fprintf(stderr, "[llm] injected %zu speech tokens from %s\n", codes.size(), codes_path.c_str());
    } else {
        sampler smp(a.seed);
        const int min_len = (int) text_tok.size() * 2;
        const int max_len = (int) text_tok.size() * 20;
        int n_past = n_prompt_rows;
        for (int i = 0; i < max_len; i++) {
            int tok = smp.sample(logits, codes, i < min_len, 6561);
            if (i >= min_len && tok >= 6561) break;
            if (tok < 0 || tok >= 6561) break;
            codes.push_back(tok);
            if (i + 1 >= max_len) break;
            std::vector<float> emb(speech_row(tok), speech_row(tok) + n_embd);
            if (decode_embd(ctx, emb.data(), 1, n_embd, n_past, true) != 0) {
                fprintf(stderr, "decode step %d failed\n", i); return 1;
            }
            n_past++;
            hidden = llama_get_embeddings_ith(ctx, -1);
            if (!hidden) { fprintf(stderr, "no hidden at step %d\n", i); return 1; }
            std::copy(hidden, hidden + n_embd, h.begin());
            if (!read_logits(logits)) {
                fprintf(stderr, "no speech logits at step %d\n", i); return 1;
            }
        }
    }
    auto t_gen1 = std::chrono::steady_clock::now();
    if (codes.empty()) { fprintf(stderr, "no speech tokens generated\n"); return 1; }
    fprintf(stderr, "[llm] generated %zu speech tokens\n", codes.size());

    if (a.dump_codes) {
        std::string path = a.out + ".codes.bin";
        FILE * f = fopen(path.c_str(), "wb");
        for (int c : codes) { int32_t v = c; fwrite(&v, 4, 1, f); }
        fclose(f);
        fprintf(stderr, "[llm] wrote %s\n", path.c_str());
    }

    // ---- flow encoder ---------------------------------------------------
    const std::vector<int> flow_codes = filter_flow_silent_tokens(codes);
    if (flow_codes.size() != codes.size()) {
        fprintf(stderr, "[flow] filtered %zu silent speech tokens (%zu -> %zu)\n",
                codes.size() - flow_codes.size(), codes.size(), flow_codes.size());
    }
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
    auto shape_of = [](const Ort::Value & v) { return v.GetTensorTypeAndShapeInfo().GetShape(); };
    auto mu_shape = shape_of(fe_out[0]);
    auto cond_shape = shape_of(fe_out[1]);
    auto spks_shape = shape_of(fe_out[2]);
    auto mask_shape = shape_of(fe_out[3]);
    if (mu_shape.size() != 3) { fprintf(stderr, "bad flow mu shape\n"); return 1; }
    const int T = (int) mu_shape[2];
    if (T != 2 * ((int) prompt_tok.size() + (int) flow_codes.size())) {
        fprintf(stderr, "flow length mismatch T=%d prompt_tok=%zu codes=%zu\n", T, prompt_tok.size(), flow_codes.size());
        return 1;
    }
    std::vector<float> mu(T * 80), cond(T * 80), spks(80), mask(T);
    memcpy(mu.data(), fe_out[0].GetTensorData<float>(), mu.size() * 4);
    memcpy(cond.data(), fe_out[1].GetTensorData<float>(), cond.size() * 4);
    memcpy(spks.data(), fe_out[2].GetTensorData<float>(), 80 * 4);
    memcpy(mask.data(), fe_out[3].GetTensorData<float>(), T * 4);
    auto t_flowenc = std::chrono::steady_clock::now();

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
    if (flow_model && T > npu_flow_cap) {
        fprintf(stderr, "flow length %d exceeds the fixed NPU capacity %d; shorten the text or use the CPU flow\\n", T, npu_flow_cap);
        return 1;
    }
    std::vector<double> ts(steps + 1);
    for (int i = 0; i <= steps; i++) ts[i] = 1.0 - std::cos(0.5 * M_PI * (double) i / steps);
    std::vector<float> x((size_t) 80 * T);
    for (int c = 0; c < 80; c++)
        for (int t = 0; t < T; t++)
            x[(size_t) c * T + t] = tab.flow_noise[(size_t) c * 15000 + t];

    std::vector<float> x_in(2 * 80 * T), mu_in(2 * 80 * T), cond_in(2 * 80 * T);
    std::vector<float> mask_in(2 * T), spks_in(2 * 80), t_in(2);
    std::vector<int64_t> x_shape{2, 80, T}, mask2_shape{2, 1, T}, spks2_shape{2, 80}, t2_shape{2};
    double t_cur = ts[0], dt = ts[1] - ts[0];
    for (int step = 1; step <= steps; step++) {
        for (int c = 0; c < 80; c++) {
            memcpy(x_in.data() + (size_t) c * T, x.data() + (size_t) c * T, T * 4);
            memcpy(x_in.data() + (size_t) (80 + c) * T, x.data() + (size_t) c * T, T * 4);
            memcpy(mu_in.data() + (size_t) c * T, mu.data() + (size_t) c * T, T * 4);
            memset(mu_in.data() + (size_t) (80 + c) * T, 0, T * 4);
            memcpy(cond_in.data() + (size_t) c * T, cond.data() + (size_t) c * T, T * 4);
            memset(cond_in.data() + (size_t) (80 + c) * T, 0, T * 4);
        }
        memcpy(mask_in.data(), mask.data(), T * 4);
        memcpy(mask_in.data() + T, mask.data(), T * 4);
        memcpy(spks_in.data(), spks.data(), 80 * 4);
        memset(spks_in.data() + 80, 0, 80 * 4);
        t_in[0] = t_in[1] = (float) t_cur;

        const float * d = nullptr;
        std::vector<Ort::Value> es_out;
        if (flow_model) {
            const size_t plane = (size_t) 80 * npu_flow_cap;
            std::vector<float> nx(2 * plane), nmu(2 * plane), ncond(2 * plane), nmask(2 * npu_flow_cap), nspks(2 * 80), nt(2);
            if (npu_flow_nlc) {
                for (int t = 0; t < T; t++) {
                    for (int c = 0; c < 80; c++) {
                        const size_t src = (size_t) c * T + t;
                        nx[(size_t) t * 80 + c] = x[src];
                        nx[plane + (size_t) t * 80 + c] = x[src];
                        // The classifier-free batch's second row is unconditional: mu and cond stay zero there,
                        // exactly as solve_euler leaves mu_in[1] / cond_in[1].  Filling them made the CFG update
                        // partly conditional and produced a systematic log-mel gain/bias.
                        nmu[(size_t) t * 80 + c] = mu[src];
                        ncond[(size_t) t * 80 + c] = cond[src];
                    }
                }
            } else {
                for (int c = 0; c < 80; c++) {
                    memcpy(nx.data() + (size_t) c * npu_flow_cap, x.data() + (size_t) c * T, T * 4);
                    memcpy(nmu.data() + (size_t) c * npu_flow_cap, mu.data() + (size_t) c * T, T * 4);
                    memcpy(ncond.data() + (size_t) c * npu_flow_cap, cond.data() + (size_t) c * T, T * 4);
                    memcpy(nx.data() + plane + (size_t) c * npu_flow_cap, x.data() + (size_t) c * T, T * 4);
                    memset(nmu.data() + plane + (size_t) c * npu_flow_cap, 0, T * 4);
                    memset(ncond.data() + plane + (size_t) c * npu_flow_cap, 0, T * 4);
                }
            }
            memcpy(nmask.data(), mask.data(), T * 4);
            memcpy(nmask.data() + npu_flow_cap, mask.data(), T * 4);
            memcpy(nspks.data(), spks.data(), 80 * 4);
            memset(nspks.data() + 80, 0, 80 * 4);
            nt[0] = nt[1] = (float) t_cur;
            if (!flow_model->set_input("x", nx.data(), nx.size() * 4) ||
                !flow_model->set_input("mask", nmask.data(), nmask.size() * 4) ||
                !flow_model->set_input("mu", nmu.data(), nmu.size() * 4) ||
                !flow_model->set_input("t", nt.data(), nt.size() * 4) ||
                !flow_model->set_input("spks", nspks.data(), nspks.size() * 4) ||
                !flow_model->set_input("cond", ncond.data(), ncond.size() * 4) || !flow_model->run()) {
                return 1;
            }
            d = flow_model->output_f32(2 * plane);
            if (!d) return 1;
        } else {
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
        const int dstride = flow_model ? npu_flow_cap : T;
        for (int c = 0; c < 80; c++) {
            for (int t = 0; t < T; t++) {
                float dphi = 1.7f * d[(size_t) c * dstride + t] - 0.7f * d[(size_t) (80 + c) * dstride + t];
                x[(size_t) c * T + t] += (float) dt * dphi;
            }
        }
        t_cur += dt;
        if (step < steps) dt = ts[step + 1] - t_cur;
    }
    auto t_flow = std::chrono::steady_clock::now();

    const int prompt_mel = 2 * (int) prompt_tok.size();
    const int Tgen = T - prompt_mel;
    std::vector<float> mel((size_t) 80 * Tgen);
    for (int c = 0; c < 80; c++)
        memcpy(mel.data() + (size_t) c * Tgen, x.data() + (size_t) c * T + prompt_mel, (size_t) Tgen * 4);
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
    if (hift_npu) {
        npu_model * hmodel = hift_npu.get();
        constexpr int cap = 512;
        // Reuse the one 512-frame vocoder for long utterances.  CausalHiFT is
        // convolution-only after the CPU front end.  Compute the source STFT
        // once on the full mel (so SineGen's phase is continuous), then give
        // each decoder window 64 frames of left context.
        constexpr int left = 64;
        constexpr int stride = cap - left;
        const bool split = !a.hift_front.empty() && !a.hift_post.empty();
        const int s_frames = cap * 480 / 4 + 1;
        double t_front = 0.0, t_npu = 0.0, t_post = 0.0;
        int chunks = 0;
        std::vector<float> full_s;
        if (split) {
            auto t0 = std::chrono::steady_clock::now();
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
    }
    auto t_hift = std::chrono::steady_clock::now();

    if (!write_wav(outpath.c_str(), pcm, 24000)) { fprintf(stderr, "failed to write %s\n", outpath.c_str()); return 1; }

    fprintf(stderr,
            "[timing] frontend %.3fs llm_load %.3fs prefill %.3fs gen %.3fs (%.1f tok/s) flow_enc %.3fs flow %.3fs hift %.3fs total %.3fs\n",
            front_s, llm_load_s,
            sec(t_req0, t_prefill), sec(t_gen0, t_gen1), codes.size() / std::max(1e-9, sec(t_gen0, t_gen1)),
            sec(t_gen1, t_flowenc), sec(t_flowenc, t_flow), sec(t_flow, t_hift),
            front_s + llm_load_s + sec(t_req0, t_hift));
    fprintf(stderr, "[audio] %.3fs %zu samples, codes=%zu flow_codes=%zu, T=%d Tgen=%d\n",
            pcm.size() / 24000.0, pcm.size(), codes.size(), flow_codes.size(), T, Tgen);
    fprintf(stderr, "[timing2] aux %.3fs sessions %.3fs npu_load %.3fs request %.3fs wall %.3fs\n",
            sec(t_aux0, t_aux1), t_sess, t_npu_load, front_s + llm_load_s + sec(t_req0, t_hift),
            sec(t_all0, std::chrono::steady_clock::now()));
    fprintf(stderr, "[out] %s\n", outpath.c_str());
    return 0;
    };

    if (a.worker) {
        // Upload weight banks and create every lazy session before the first
        // request, so service start time (not first-token latency) pays for it.
        if (!a.prompt_wav.empty()) {
            auto warm = prepare_ref(a.prompt_wav);
            if (warm) {
                generate("你好。", a.prompt_text, "/tmp/cosy_worker_warmup.wav", *warm,
                         std::string(), 0.0, 0.0);
                unlink("/tmp/cosy_worker_warmup.wav");
            }
        }
        FILE * in = stdin;
        for (;;) {
            uint32_t lens[4] = {};
            if (fread(lens, sizeof(uint32_t), 4, in) != 4) break;
            std::string fields[4];
            bool ok = true;
            for (int i = 0; i < 4; i++) {
                if (lens[i] > (1u << 20)) { ok = false; break; }
                fields[i].resize(lens[i]);
                if (lens[i] && fread(fields[i].data(), 1, lens[i], in) != lens[i]) { ok = false; break; }
            }
            if (!ok) break;
            auto t_front0 = std::chrono::steady_clock::now();
            auto rf = prepare_ref(fields[2]);
            double front_s = sec(t_front0, std::chrono::steady_clock::now());
            int rc = rf ? generate(fields[0], fields[1], fields[3], *rf, std::string(), front_s, 0.0) : 1;
            fprintf(stdout, rc == 0 ? "OK\n" : "ERR\n");
            fflush(stdout);
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
        if (generate(a.text, a.prompt_text, outpath, *rf0,
                     rep == 0 ? a.codes_in : std::string(),
                     rep == 0 ? front_s : 0.0, rep == 0 ? llm_load_s : 0.0) != 0) return 1;
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
