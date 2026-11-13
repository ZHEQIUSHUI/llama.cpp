// Fun-CosyVoice3-0.5B-2512 end-to-end runner.
//
// The Qwen2 backbone is evaluated through llama.cpp (CPU or the AXERA backend).
// The reference-audio encoders, flow-matching decoder and HiFT vocoder use the
// official ONNX graphs through ONNX Runtime on the host.  The speech token
// table/head are stored in a small auxiliary GGUF next to the backbone GGUF.

#include "llama.h"
#include "llama-ext.h"
#include "ggml.h"
#include "gguf.h"
#include "mtmd-audio.h"

#include "onnxruntime_cxx_api.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
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

static std::vector<float> load_f32(const std::string & path, size_t expected = 0) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    size_t n = (size_t) f.tellg() / sizeof(float);
    if (expected && n != expected) { fprintf(stderr, "%s: expected %zu floats, got %zu\n", path.c_str(), expected, n); std::exit(1); }
    std::vector<float> v(n); f.seekg(0); f.read((char *) v.data(), n * sizeof(float)); return v;
}

// ---------------------------------------------------------------------------
// ONNX Runtime helpers

struct ort_session {
    Ort::Session session{nullptr};
    std::vector<std::string> in_names, out_names;
    std::vector<const char *> in_c, out_c;

    ort_session() = default;
    ort_session(Ort::Env & env, const std::string & path) {
        Ort::SessionOptions opt;
        opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
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
    std::vector<float> frame(n_fft);
    float vmax = -1e30f;
    for (int i = 0; i < n_frames; i++) {
        std::copy(x.begin() + (size_t) i * hop, x.begin() + (size_t) i * hop + n_fft, frame.begin());
        float * dst = mel.data() + (size_t) i * 128;
        mtmd_audio_log_mel_frame(cache, frame.data(), n_fft, 128, dst);
        for (int j = 0; j < 128; j++) vmax = std::max(vmax, dst[j]);
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
        for (int j = 0; j < 80; j++) { dst[j] *= 2.302585092994046f; mean[j] += dst[j]; }
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
    cache.filters.n_mel = 80;
    cache.filters.n_fft = 1920;
    cache.filters.data = t.flow_mel;

    const int n_fft = 1920, hop = 480, pad = (n_fft - hop) / 2;
    std::vector<float> x(pcm24.size() + 2 * pad);
    for (int i = 0; i < pad; i++) x[i] = pcm24[pad - i];
    std::copy(pcm24.begin(), pcm24.end(), x.begin() + pad);
    for (int i = 0; i < pad; i++) x[pcm24.size() + pad + i] = pcm24[pcm24.size() - 2 - i];

    n_frames = (int) (pcm24.size() / hop);
    std::vector<float> feat((size_t) n_frames * 80);
    std::vector<float> frame(n_fft);
    const float floor_log = std::log(1e-5f);
    for (int i = 0; i < n_frames; i++) {
        std::copy(x.begin() + (size_t) i * hop, x.begin() + (size_t) i * hop + n_fft, frame.begin());
        float * dst = feat.data() + (size_t) i * 80;
        mtmd_audio_log_mel_frame(cache, frame.data(), n_fft, 80, dst);
        for (int j = 0; j < 80; j++) {
            // helper: log10(power); official: log(max(sqrt(power), 1e-5)).
            dst[j] = std::max(0.5f * 2.302585092994046f * dst[j], floor_log);
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
// Sampling: nucleus(top_p=0.8, top_k=25) with repetition-aware resampling.

struct sampler {
    std::mt19937 rng;
    explicit sampler(uint32_t seed) : rng(seed) {}

    int sample(std::vector<float> logits, const std::vector<int> & decoded, bool ignore_eos, int eos, int min_len) {
        if (ignore_eos && eos >= 0 && eos < (int) logits.size()) logits[eos] = -1e30f;
        float mx = *std::max_element(logits.begin(), logits.end());
        double sum = 0;
        for (auto & v : logits) { v = std::exp(v - mx); sum += v; }
        for (auto & v : logits) v /= (float) sum;

        std::vector<int> idx(logits.size());
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + std::min<size_t>(25, idx.size()), idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        std::vector<double> p;
        double cum = 0;
        for (int j = 0; j < (int) idx.size() && j < 25; j++) {
            p.push_back(logits[idx[j]]);
            cum += logits[idx[j]];
            if (cum >= 0.8) break;
        }
        std::discrete_distribution<int> dist(p.begin(), p.end());
        int top = idx[dist(rng)];

        int rep = 0;
        for (int i = std::max(0, (int) decoded.size() - 10); i < (int) decoded.size(); i++) rep += decoded[i] == top;
        if (rep >= 1) {
            std::discrete_distribution<int> all(logits.begin(), logits.end());
            top = all(rng);
        }
        (void) min_len;
        return top;
    }
};

// ---------------------------------------------------------------------------
// Main

struct args {
    std::string model, aux, tables;
    std::string campplus, tokenizer, flow_encoder, flow_estimator, hift;
    std::string text, prompt_text, prompt_wav, out = "out.wav";
    int ngl = 0, ctx = 4096, threads = 16;
    uint32_t seed = 1986;
    bool dump_codes = false;
    std::string parity_in, parity_out, dump_feats;
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
        else if (k == "-p" || k == "--text") a.text = val();
        else if (k == "--prompt-text") a.prompt_text = val();
        else if (k == "--prompt-wav") a.prompt_wav = val();
        else if (k == "-o" || k == "--out") a.out = val();
        else if (k == "-ngl") a.ngl = std::stoi(val());
        else if (k == "-c") a.ctx = std::stoi(val());
        else if (k == "-t") a.threads = std::stoi(val());
        else if (k == "--seed") a.seed = (uint32_t) std::stoul(val());
        else if (k == "--dump-codes") a.dump_codes = true;
        else if (k == "--parity-in") a.parity_in = val();
        else if (k == "--parity-out") a.parity_out = val();
        else if (k == "--dump-feats") a.dump_feats = val();
        else { fprintf(stderr, "unknown option %s\n", k.c_str()); return false; }
    }
    return !a.model.empty() && !a.aux.empty() && !a.tables.empty() && !a.text.empty() && !a.prompt_wav.empty() &&
           !a.campplus.empty() && !a.tokenizer.empty() && !a.flow_encoder.empty() &&
           !a.flow_estimator.empty() && !a.hift.empty();
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
            "  -p TEXT --prompt-text TEXT --prompt-wav ref.wav -o out.wav [-ngl N] [--dump-codes]\n", argv[0]);
        return 1;
    }

    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    auto t_all0 = std::chrono::steady_clock::now();
    auto t0 = t_all0;

    feature_tables tab = load_tables(a.tables);
    aux_weights aux;
    if (!load_aux(a.aux, aux)) return 1;

    wav_data wav;
    if (!read_wav(a.prompt_wav.c_str(), wav)) return 1;
    auto pcm16 = resample_linear(wav.pcm, wav.sample_rate, 16000);
    auto pcm24 = resample_linear(wav.pcm, wav.sample_rate, 24000);

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "cosyvoice3");
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto make_session = [&](const std::string & path) {
        return std::make_unique<ort_session>(env, path);
    };
    auto campplus = make_session(a.campplus);
    auto tokenizer = make_session(a.tokenizer);
    auto flow_enc = make_session(a.flow_encoder);
    auto estimator = make_session(a.flow_estimator);
    auto hift = make_session(a.hift);

    // ---- reference audio front ends -------------------------------------
    int fbank_frames = 0, wmel_frames = 0, flow_frames = 0;
    auto fbank = kaldi_fbank80(pcm16, fbank_frames, tab);
    std::vector<int64_t> spk_shape{1, fbank_frames, 80};
    Ort::Value spk_in = Ort::Value::CreateTensor<float>(mem, fbank.data(), fbank.size(), spk_shape.data(), spk_shape.size());
    std::vector<Ort::Value> spk_inputs;
    spk_inputs.emplace_back(std::move(spk_in));
    auto spk_out = campplus->run(std::move(spk_inputs));
    const float * spk_data = spk_out[0].GetTensorData<float>();
    std::vector<float> embedding(spk_data, spk_data + 192);

    auto wmel = whisper_mel(pcm16, wmel_frames, tab);
    std::vector<int64_t> wmel_shape{1, 128, wmel_frames};
    std::vector<int32_t> wmel_len{wmel_frames};
    std::vector<int64_t> one_shape{1};
    std::vector<Ort::Value> tok_in;
    tok_in.emplace_back(Ort::Value::CreateTensor<float>(mem, wmel.data(), wmel.size(), wmel_shape.data(), wmel_shape.size()));
    tok_in.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, wmel_len.data(), 1, one_shape.data(), one_shape.size()));
    auto tok_out = tokenizer->run(std::move(tok_in));
    const int32_t * tok_data = tok_out[0].GetTensorData<int32_t>();
    auto tok_info = tok_out[0].GetTensorTypeAndShapeInfo();
    size_t n_prompt_tok = 1;
    for (auto d : tok_info.GetShape()) n_prompt_tok *= (size_t) d;
    std::vector<int32_t> prompt_tok(tok_data, tok_data + n_prompt_tok);

    auto prompt_feat = flow_mel24(pcm24, flow_frames, tab);
    fprintf(stderr, "[frontend] fbank=%d whisper_mel=%d flow_mel=%d speech_tokens=%zu\n",
            fbank_frames, wmel_frames, flow_frames, prompt_tok.size());
    if (!a.dump_feats.empty()) {
        auto dump = [&](const std::string & suffix, const void * data, size_t bytes) {
            FILE * f = fopen((a.dump_feats + suffix).c_str(), "wb"); fwrite(data, 1, bytes, f); fclose(f);
        };
        dump(".fbank.f32", fbank.data(), fbank.size() * 4);
        dump(".wmel.f32", wmel.data(), wmel.size() * 4);
        dump(".ptok.i32", prompt_tok.data(), prompt_tok.size() * 4);
        dump(".embedding.f32", embedding.data(), embedding.size() * 4);
        dump(".flowfeat.f32", prompt_feat.data(), prompt_feat.size() * 4);
        FILE * f = fopen((a.dump_feats + ".shape.txt").c_str(), "w");
        fprintf(f, "fbank %d 80\nwmel 128 %d\nptok %zu\nflowfeat %d 80\n", fbank_frames, wmel_frames, prompt_tok.size(), flow_frames);
        fclose(f);
    }
    auto t_front = std::chrono::steady_clock::now();

    // ---- Qwen2 backbone -------------------------------------------------
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
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "failed to create llama context\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_embd = llama_model_n_embd(model);
    if (n_embd != aux.n_embd) { fprintf(stderr, "embedding dim mismatch %d vs %d\n", n_embd, aux.n_embd); return 1; }

    if (!a.parity_in.empty()) {
        auto in = load_f32(a.parity_in);
        int rows = (int) (in.size() / n_embd);
        if (decode_embd(ctx, in.data(), rows, n_embd, 0, true) != 0) { fprintf(stderr, "parity decode failed\n"); return 1; }
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
        dump(a.parity_out + ".logits.f32", logits.data(), logits.size());
        fprintf(stderr, "parity rows=%d hidden=%d logits=%d\n", rows, n_embd, aux.n_speech);
        llama_free(ctx); llama_model_free(model); llama_backend_free();
        return 0;
    }

    auto tokenize = [&](const std::string & text) {
        std::vector<llama_token> v(text.size() + 64);
        int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), v.data(), (int) v.size(), false, true);
        if (n < 0) { v.resize(-n); n = llama_tokenize(vocab, text.c_str(), (int) text.size(), v.data(), (int) v.size(), false, true); }
        if (n < 0) { fprintf(stderr, "tokenize failed\n"); std::exit(1); }
        v.resize(n);
        return v;
    };
    auto prompt_text_tok = tokenize(a.prompt_text);
    auto text_tok = tokenize(a.text);
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

    if (decode_embd(ctx, lm_input.data(), n_prompt_rows, n_embd, 0, true) != 0) {
        fprintf(stderr, "prefill decode failed\n"); return 1;
    }
    const float * hidden = llama_get_embeddings_ith(ctx, -1);
    if (!hidden) { fprintf(stderr, "no hidden state\n"); return 1; }
    std::vector<float> h(hidden, hidden + n_embd);
    auto t_prefill = std::chrono::steady_clock::now();

    // ---- autoregressive speech token sampling ---------------------------
    std::vector<float> logits(aux.n_speech);
    auto compute_logits = [&](const std::vector<float> & hidden_vec, std::vector<float> & out) {
        for (int j = 0; j < aux.n_speech; j++) {
            const float * w = aux.speech_dec.data() + (size_t) j * n_embd;
            double s = 0;
            for (int k = 0; k < n_embd; k++) s += (double) w[k] * hidden_vec[k];
            out[j] = (float) s;
        }
    };
    sampler smp(a.seed);
    std::vector<int> codes;
    const int min_len = (int) text_tok.size() * 2;
    const int max_len = (int) text_tok.size() * 20;
    int n_past = n_prompt_rows;
    auto t_gen0 = std::chrono::steady_clock::now();
    for (int i = 0; i < max_len; i++) {
        compute_logits(h, logits);
        int tok = smp.sample(logits, codes, i < min_len, 6561, min_len);
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
    std::vector<int32_t> gen_tok(codes.begin(), codes.end());
    std::vector<int64_t> tok_shape{1, (int64_t) gen_tok.size()};
    std::vector<int64_t> ptok_shape{1, (int64_t) prompt_tok.size()};
    std::vector<int64_t> pfeat_shape{1, flow_frames, 80};
    std::vector<int64_t> emb_shape{1, 192};
    std::vector<Ort::Value> fe_in;
    fe_in.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, gen_tok.data(), gen_tok.size(), tok_shape.data(), tok_shape.size()));
    fe_in.emplace_back(Ort::Value::CreateTensor<int32_t>(mem, prompt_tok.data(), prompt_tok.size(), ptok_shape.data(), ptok_shape.size()));
    fe_in.emplace_back(Ort::Value::CreateTensor<float>(mem, prompt_feat.data(), prompt_feat.size(), pfeat_shape.data(), pfeat_shape.size()));
    fe_in.emplace_back(Ort::Value::CreateTensor<float>(mem, embedding.data(), embedding.size(), emb_shape.data(), emb_shape.size()));
    auto fe_out = flow_enc->run(std::move(fe_in));
    auto shape_of = [](const Ort::Value & v) { return v.GetTensorTypeAndShapeInfo().GetShape(); };
    auto mu_shape = shape_of(fe_out[0]);
    auto cond_shape = shape_of(fe_out[1]);
    auto spks_shape = shape_of(fe_out[2]);
    auto mask_shape = shape_of(fe_out[3]);
    if (mu_shape.size() != 3) { fprintf(stderr, "bad flow mu shape\n"); return 1; }
    const int T = (int) mu_shape[2];
    if (T != 2 * ((int) prompt_tok.size() + (int) codes.size())) {
        fprintf(stderr, "flow length mismatch T=%d prompt_tok=%zu codes=%zu\n", T, prompt_tok.size(), codes.size());
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

        std::vector<Ort::Value> es_in;
        es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, x_in.data(), x_in.size(), x_shape.data(), x_shape.size()));
        es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, mask_in.data(), mask_in.size(), mask2_shape.data(), mask2_shape.size()));
        es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, mu_in.data(), mu_in.size(), x_shape.data(), x_shape.size()));
        es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, t_in.data(), 2, t2_shape.data(), t2_shape.size()));
        es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, spks_in.data(), spks_in.size(), spks2_shape.data(), spks2_shape.size()));
        es_in.emplace_back(Ort::Value::CreateTensor<float>(mem, cond_in.data(), cond_in.size(), x_shape.data(), x_shape.size()));
        auto es_out = estimator->run(std::move(es_in));
        const float * d = es_out[0].GetTensorData<float>();
        for (int c = 0; c < 80; c++) {
            for (int t = 0; t < T; t++) {
                float dphi = 1.7f * d[(size_t) c * T + t] - 0.7f * d[(size_t) (80 + c) * T + t];
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
    std::vector<int64_t> hmel_shape{1, 80, Tgen};
    std::vector<Ort::Value> hi_in;
    hi_in.emplace_back(Ort::Value::CreateTensor<float>(mem, mel.data(), mel.size(), hmel_shape.data(), hmel_shape.size()));
    auto hi_out = hift->run(std::move(hi_in));
    auto audio_shape = shape_of(hi_out[0]);
    const float * audio = hi_out[0].GetTensorData<float>();
    size_t n_samples = 1;
    for (auto d : audio_shape) n_samples *= (size_t) d;
    std::vector<float> pcm(audio, audio + n_samples);
    auto t_hift = std::chrono::steady_clock::now();

    if (!write_wav(a.out.c_str(), pcm, 24000)) { fprintf(stderr, "failed to write %s\n", a.out.c_str()); return 1; }

    auto sec = [](auto a0, auto a1) { return std::chrono::duration<double>(a1 - a0).count(); };
    fprintf(stderr,
            "[timing] frontend %.3fs llm_load %.3fs prefill %.3fs gen %.3fs (%.1f tok/s) flow_enc %.3fs flow %.3fs hift %.3fs total %.3fs\n",
            sec(t0, t_front), 0.0,
            sec(t_front, t_prefill), sec(t_gen0, t_gen1), codes.size() / std::max(1e-9, sec(t_gen0, t_gen1)),
            sec(t_gen1, t_flowenc), sec(t_flowenc, t_flow), sec(t_flow, t_hift), sec(t_all0, t_hift));
    fprintf(stderr, "[audio] %.3fs %zu samples, codes=%zu, T=%d Tgen=%d\n", n_samples / 24000.0, n_samples, codes.size(), T, Tgen);
    fprintf(stderr, "[out] %s\n", a.out.c_str());

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
