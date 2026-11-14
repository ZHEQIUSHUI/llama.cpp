#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct options {
    std::string transcript;
    std::string language = "English";
    std::string output = "-";
    int segment_ms = 80;
};

struct seq_item {
    bool embd = false;
    llama_token token = LLAMA_TOKEN_NULL;
    size_t audio_row = 0;
};

struct word_span {
    std::string text;
    double start = 0.0;
    double end = 0.0;
};

static bool is_space(uint32_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == 0x3000;
}

static bool is_cjk(uint32_t c) {
    return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
           (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF) ||
           (c >= 0xF900 && c <= 0xFAFF);
}

static bool is_kept(uint32_t c) {
    if (c == '\'') {
        return true;
    }
    if ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || is_cjk(c)) {
        return true;
    }
    if (c < 0x80) {
        return false;
    }
    // Approximate Unicode L/N: keep letters and numbers, drop the common CJK and general punctuation blocks.
    if ((c >= 0x2000 && c <= 0x206F) || (c >= 0x3000 && c <= 0x303F) ||
        (c >= 0xFE10 && c <= 0xFE1F) || (c >= 0xFE30 && c <= 0xFE4F) ||
        (c >= 0xFF00 && c <= 0xFF0F) || (c >= 0xFF1A && c <= 0xFF20) ||
        (c >= 0xFF3B && c <= 0xFF40) || (c >= 0xFF5B && c <= 0xFF65)) {
        return false;
    }
    return true;
}

static std::vector<uint32_t> utf8_decode(const std::string & s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        uint32_t cp = 0;
        int n = 0;
        if (c < 0x80) {
            cp = c;
            n = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            n = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            n = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            n = 4;
        } else {
            throw std::runtime_error("invalid UTF-8");
        }
        if (i + n > s.size()) {
            throw std::runtime_error("truncated UTF-8");
        }
        for (int k = 1; k < n; ++k) {
            cp = (cp << 6) | ((unsigned char) s[i + k] & 0x3F);
        }
        out.push_back(cp);
        i += n;
    }
    return out;
}

static std::string utf8_encode(uint32_t c) {
    std::string s;
    if (c < 0x80) {
        s.push_back((char) c);
    } else if (c < 0x800) {
        s.push_back((char) (0xC0 | (c >> 6)));
        s.push_back((char) (0x80 | (c & 0x3F)));
    } else if (c < 0x10000) {
        s.push_back((char) (0xE0 | (c >> 12)));
        s.push_back((char) (0x80 | ((c >> 6) & 0x3F)));
        s.push_back((char) (0x80 | (c & 0x3F)));
    } else {
        s.push_back((char) (0xF0 | (c >> 18)));
        s.push_back((char) (0x80 | ((c >> 12) & 0x3F)));
        s.push_back((char) (0x80 | ((c >> 6) & 0x3F)));
        s.push_back((char) (0x80 | (c & 0x3F)));
    }
    return s;
}

static std::vector<std::string> split_words(const std::string & text) {
    std::vector<std::string> words;
    std::string clean;
    std::vector<uint32_t> segment;
    auto flush_segment = [&]() {
        if (segment.empty()) {
            return;
        }
        // split_segment_with_chinese: CJK characters are words, the rest stays grouped
        std::string latin;
        for (uint32_t cp : segment) {
            if (is_cjk(cp)) {
                if (!latin.empty()) {
                    words.push_back(latin);
                    latin.clear();
                }
                words.push_back(utf8_encode(cp));
            } else {
                latin += utf8_encode(cp);
            }
        }
        if (!latin.empty()) {
            words.push_back(latin);
        }
        segment.clear();
    };
    for (uint32_t cp : utf8_decode(text)) {
        if (is_space(cp)) {
            flush_segment();
            continue;
        }
        if (is_kept(cp)) {
            segment.push_back(cp);
        } else {
            flush_segment();
        }
    }
    flush_segment();
    return words;
}

static std::string timestamp_prompt(const std::vector<std::string> & words) {
    std::string s;
    for (size_t i = 0; i < words.size(); ++i) {
        if (i) {
            s += "<timestamp><timestamp>";
        }
        s += words[i];
    }
    s += "<timestamp><timestamp>";
    return s;
}

static int64_t floor_div(int64_t a, int64_t b) {
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r < 0) != (b < 0))) {
        --q;
    }
    return q;
}

static int64_t feat_output_length(int64_t length) {
    const int64_t leave = length % 100;
    const int64_t feat = floor_div(leave - 1, 2) + 1;
    return floor_div(floor_div(feat - 1, 2) + 1 - 1, 2) + 1 + (length / 100) * 13;
}

static std::vector<int64_t> fix_timestamp(const std::vector<int64_t> & data) {
    const int n = (int) data.size();
    if (n == 0) {
        return {};
    }
    std::vector<int> dp(n, 1), parent(n, -1);
    for (int i = 1; i < n; ++i) {
        for (int j = 0; j < i; ++j) {
            if (data[j] <= data[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1;
                parent[i] = j;
            }
        }
    }
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (dp[i] > dp[best]) {
            best = i;
        }
    }
    std::vector<char> normal(n, 0);
    for (int i = best; i >= 0; i = parent[i]) {
        normal[i] = 1;
    }
    std::vector<int64_t> result = data;
    for (int i = 0; i < n;) {
        if (normal[i]) {
            ++i;
            continue;
        }
        int j = i;
        while (j < n && !normal[j]) {
            ++j;
        }
        const int count = j - i;
        int64_t left = 0, right = 0;
        bool has_left = false, has_right = false;
        for (int k = i - 1; k >= 0; --k) {
            if (normal[k]) {
                left = result[k];
                has_left = true;
                break;
            }
        }
        for (int k = j; k < n; ++k) {
            if (normal[k]) {
                right = result[k];
                has_right = true;
                break;
            }
        }
        if (count <= 2) {
            for (int k = i; k < j; ++k) {
                if (!has_left) {
                    result[k] = right;
                } else if (!has_right) {
                    result[k] = left;
                } else {
                    result[k] = (k - (i - 1)) <= (j - k) ? left : right;
                }
            }
        } else if (has_left && has_right) {
            const double step = (double) (right - left) / (count + 1);
            for (int k = i; k < j; ++k) {
                result[k] = (int64_t) (left + step * (k - i + 1));
            }
        } else if (has_left) {
            for (int k = i; k < j; ++k) {
                result[k] = left;
            }
        } else if (has_right) {
            for (int k = i; k < j; ++k) {
                result[k] = right;
            }
        }
        i = j;
    }
    return result;
}

static void json_string(std::string & out, const std::string & s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += (char) c;
                }
        }
    }
    out += '"';
}

static std::string take_value(int & i, int argc, char ** argv, const char * name) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
    }
    return argv[++i];
}

static std::vector<std::string> strip_custom_args(int & argc, char ** argv, options & opt) {
    std::vector<std::string> args;
    args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--transcript" || a == "--text") {
            opt.transcript = take_value(i, argc, argv, a.c_str());
        } else if (a == "--language") {
            opt.language = take_value(i, argc, argv, a.c_str());
        } else if (a == "--output-json" || a == "--output") {
            opt.output = take_value(i, argc, argv, a.c_str());
        } else if (a == "--segment-ms") {
            opt.segment_ms = std::stoi(take_value(i, argc, argv, a.c_str()));
        } else {
            args.push_back(a);
        }
    }
    return args;
}

static void print_usage(int argc, char ** argv) {
    (void) argc;
    const char * argv0 = argv[0];
    fprintf(stderr,
            "usage: %s -m model.gguf --mmproj mmproj.gguf --audio audio.wav --transcript TEXT [options]\n"
            "  --language NAME       Chinese or English (default: English)\n"
            "  --output-json PATH    JSON output path, - for stdout (default: -)\n"
            "  --segment-ms N        timestamp step in ms (default: 80)\n"
            "  -c N -ub N -ngl N     standard llama.cpp options are accepted\n",
            argv0);
}

static int argmax_class(const float * row, int n_vocab, int n_class, int * margin2) {
    int first = n_vocab - n_class;
    while (first < n_vocab && !std::isfinite(row[first])) {
        ++first;
    }
    if (first >= n_vocab) {
        return -1;
    }
    int best = first;
    double second = -INFINITY;
    for (int i = first + 1; i < n_vocab; ++i) {
        if (!std::isfinite(row[i])) {
            continue;
        }
        if (row[i] > row[best]) {
            second = row[best];
            best = i;
        } else if (row[i] > second) {
            second = row[i];
        }
    }
    if (margin2) {
        *margin2 = (int) ((row[best] - second) * 1e6f);
    }
    return best - (n_vocab - n_class);
}

} // namespace

int main(int argc, char ** argv) {
    options opt;
    try {
        std::vector<std::string> args = strip_custom_args(argc, argv, opt);
        if (opt.transcript.empty()) {
            print_usage(argc, argv);
            return 1;
        }
        std::vector<char *> av;
        for (auto & s : args) {
            av.push_back(s.data());
        }

        common_params params;
        common_init();
        if (!common_params_parse((int) av.size(), av.data(), params, LLAMA_EXAMPLE_MTMD, print_usage)) {
            return 1;
        }
        if (params.mmproj.path.empty() || params.image.empty()) {
            print_usage(argc, argv);
            fprintf(stderr, "error: --mmproj and --audio are required\n");
            return 1;
        }
        if (opt.language != "Chinese" && opt.language != "English") {
            fprintf(stderr, "error: language %s is not implemented (Chinese and English only)\n", opt.language.c_str());
            return 1;
        }
        if (params.n_ctx <= 0) {
            params.n_ctx = 4096;
        }
        params.n_ctx = std::min(params.n_ctx, 4096);
        params.n_batch = std::max(params.n_batch, 4096);
        params.n_ubatch = 256;
        params.warmup = false;

        ggml_backend_load_all();
        common_init_result_ptr init = common_init_from_params(params);
        llama_model * model = init->model();
        llama_context * ctx = init->context();
        if (!model || !ctx) {
            fprintf(stderr, "error: model/context init failed\n");
            return 1;
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const int n_class = llama_model_n_embd_out(model);
        const int n_embd = llama_model_n_embd_inp(model);
        if (n_class <= 0 || n_class > n_vocab) {
            fprintf(stderr, "error: invalid classifier width %d (tokenizer vocabulary %d)\n", n_class, n_vocab);
            return 1;
        }

        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.device = params.mmproj_device;
        mparams.print_timings = true;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup = false;
        mtmd_context * mctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams);
        if (!mctx) {
            fprintf(stderr, "error: failed to load mmproj\n");
            return 1;
        }

        const std::string audio_path = params.image[0];
        mtmd_helper_bitmap_wrapper bm = mtmd_helper_bitmap_init_from_file(
                mctx, audio_path.c_str(), false, mtmd_helper_init_opt_default());
        if (!bm.bitmap || !mtmd_bitmap_is_audio(bm.bitmap)) {
            fprintf(stderr, "error: failed to load audio %s\n", audio_path.c_str());
            return 1;
        }
        const int64_t n_samples = (int64_t) mtmd_bitmap_get_nx(bm.bitmap);

        std::vector<std::string> word_list = split_words(opt.transcript);
        if (word_list.empty()) {
            fprintf(stderr, "error: transcript has no alignable words\n");
            return 1;
        }
        const std::string suffix = timestamp_prompt(word_list);

        mtmd_input_text text_empty = { "", 0, false, true };
        mtmd_input_text text_suffix = { suffix.data(), suffix.size(), false, true };
        mtmd_input_part parts[3] = {
            { &text_empty, nullptr },
            { nullptr, bm.bitmap },
            { &text_suffix, nullptr },
        };
        const mtmd_input_part * part_ptrs[3] = { &parts[0], &parts[1], &parts[2] };
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
                mtmd_input_chunks_init(), mtmd_input_chunks_free);
        if (mtmd_tokenize_from_parts(mctx, chunks.get(), part_ptrs, 3, false) != 0) {
            fprintf(stderr, "error: tokenize failed\n");
            return 1;
        }

        size_t ts_id = (size_t) -1;
        {
            std::vector<llama_token> t;
            t.resize(32);
            int n = llama_tokenize(vocab, "<timestamp>", 11, t.data(), (int) t.size(), false, true);
            if (n != 1) {
                fprintf(stderr, "error: <timestamp> did not tokenize to one token (n=%d)\n", n);
                return 1;
            }
            ts_id = (size_t) t[0];
        }

        const int64_t total_frames = n_samples / 160;
        int64_t window_off = 0;
        size_t audio_chunk_index = 0;
        int64_t official_audio_tokens = 0;
        std::vector<seq_item> items;
        std::vector<std::vector<float>> audio_rows;
        std::vector<size_t> timestamp_items;

        const size_t n_chunks = mtmd_input_chunks_size(chunks.get());
        for (size_t ci = 0; ci < n_chunks; ++ci) {
            const mtmd_input_chunk * chunk = mtmd_input_chunks_get(chunks.get(), ci);
            const mtmd_input_chunk_type type = mtmd_input_chunk_get_type(chunk);
            if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                size_t nt = 0;
                const llama_token * toks = mtmd_input_chunk_get_tokens_text(chunk, &nt);
                for (size_t i = 0; i < nt; ++i) {
                    seq_item item;
                    item.token = toks[i];
                    if (toks[i] == (llama_token) ts_id) {
                        timestamp_items.push_back(items.size());
                    }
                    items.push_back(item);
                }
            } else if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
                const int64_t win_eff = std::min<int64_t>(800, total_frames - window_off);
                if (win_eff <= 0) {
                    fprintf(stderr, "error: more audio chunks than mel windows\n");
                    return 1;
                }
                const int64_t want = feat_output_length(win_eff);
                const size_t got = mtmd_input_chunk_get_n_tokens(chunk);
                if ((int64_t) got < want) {
                    fprintf(stderr, "error: audio chunk %zu has %zu rows, official window needs %lld\n",
                            audio_chunk_index, got, (long long) want);
                    return 1;
                }
                if (mtmd_encode_chunk(mctx, chunk) != 0) {
                    fprintf(stderr, "error: audio encode failed\n");
                    return 1;
                }
                const float * embd = mtmd_get_output_embd(mctx);
                if (!embd) {
                    fprintf(stderr, "error: no audio embeddings\n");
                    return 1;
                }
                for (int64_t r = 0; r < want; ++r) {
                    seq_item item;
                    item.embd = true;
                    item.audio_row = audio_rows.size();
                    audio_rows.emplace_back(embd + (size_t) r * n_embd, embd + (size_t) (r + 1) * n_embd);
                    items.push_back(item);
                }
                official_audio_tokens += want;
                window_off += win_eff;
                ++audio_chunk_index;
            }
        }
        if (window_off < total_frames) {
            fprintf(stderr, "error: audio chunks cover %lld of %lld mel frames\n",
                    (long long) window_off, (long long) total_frames);
            return 1;
        }
        if (timestamp_items.size() != word_list.size() * 2) {
            fprintf(stderr, "error: %zu timestamp tokens for %zu words (expected %zu)\n",
                    timestamp_items.size(), word_list.size(), word_list.size() * 2);
            return 1;
        }
        if (items.size() > (size_t) params.n_batch) {
            fprintf(stderr, "error: prompt has %zu tokens, n_batch is %d (raise -b)\n", items.size(), params.n_batch);
            return 1;
        }
        if ((int64_t) items.size() > params.n_ctx) {
            fprintf(stderr, "error: prompt has %zu tokens, context is %d\n", items.size(), params.n_ctx);
            return 1;
        }

        std::unique_ptr<llama_batch_ext, decltype(&llama_batch_ext_free)> batch(
                llama_batch_ext_init(ctx), llama_batch_ext_free);
        for (size_t i = 0; i < items.size(); ++i) {
            const seq_item & item = items[i];
            int32_t idx = -1;
            if (item.embd) {
                llama_embd e = { audio_rows[item.audio_row].data(), 1, (size_t) n_embd };
                idx = llama_batch_ext_add_embd(batch.get(), 0, e);
            } else {
                idx = llama_batch_ext_add_token(batch.get(), 0, item.token);
            }
            if (idx < 0) {
                fprintf(stderr, "error: failed to add batch item %zu\n", i);
                return 1;
            }
            llama_pos p[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, (llama_pos) i };
            if (!llama_batch_ext_set_pos(batch.get(), idx, p)) {
                fprintf(stderr, "error: failed to set position %zu\n", i);
                return 1;
            }
        }
        for (size_t idx : timestamp_items) {
            llama_batch_ext_set_output_logits(batch.get(), (int32_t) idx, true);
        }

        const int64_t t_forward_us = ggml_time_us();
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            fprintf(stderr, "error: llama_process failed\n");
            return 1;
        }
        const double forward_s = (ggml_time_us() - t_forward_us) / 1e6;

        std::vector<int64_t> raw_ms;
        std::vector<int> argmax_ids;
        std::vector<int> margins;
        raw_ms.reserve(timestamp_items.size());
        argmax_ids.reserve(timestamp_items.size());
        margins.reserve(timestamp_items.size());
        for (size_t idx : timestamp_items) {
            const float * row = llama_get_logits_ith(ctx, (int32_t) idx);
            if (!row) {
                fprintf(stderr, "error: no logits at token %zu\n", idx);
                return 1;
            }
            int margin = 0;
            const int id = argmax_class(row, n_vocab, n_class, &margin);
            if (id < 0 || id >= n_class) {
                fprintf(stderr, "error: invalid argmax %d\n", id);
                return 1;
            }
            argmax_ids.push_back(id);
            margins.push_back(margin);
            raw_ms.push_back((int64_t) id * opt.segment_ms);
        }
        const std::vector<int64_t> fixed = fix_timestamp(raw_ms);

        std::vector<word_span> spans;
        spans.reserve(word_list.size());
        for (size_t i = 0; i < word_list.size(); ++i) {
            word_span s;
            s.text = word_list[i];
            s.start = std::round((double) fixed[2 * i] / 1000.0 * 1000.0) / 1000.0;
            s.end = std::round((double) fixed[2 * i + 1] / 1000.0 * 1000.0) / 1000.0;
            spans.push_back(std::move(s));
        }

        std::string js;
        js += "{\n  \"task\": \"align\",\n  \"language\": ";
        json_string(js, opt.language);
        js += ",\n  \"duration\": " + std::to_string((double) n_samples / 16000.0);
        js += ",\n  \"text\": ";
        json_string(js, opt.transcript);
        js += ",\n  \"words\": [";
        for (size_t i = 0; i < spans.size(); ++i) {
            if (i) js += ",";
            js += "\n    {\"word\": ";
            json_string(js, spans[i].text);
            char buf[128];
            snprintf(buf, sizeof(buf), ", \"start\": %.3f, \"end\": %.3f}", spans[i].start, spans[i].end);
            js += buf;
        }
        js += "\n  ],\n  \"timestamp_ids\": [";
        for (size_t i = 0; i < argmax_ids.size(); ++i) {
            if (i) js += ", ";
            js += std::to_string(argmax_ids[i]);
        }
        js += "],\n  \"raw_timestamp_ms\": [";
        for (size_t i = 0; i < raw_ms.size(); ++i) {
            if (i) js += ", ";
            js += std::to_string(raw_ms[i]);
        }
        js += "],\n  \"fixed_timestamp_ms\": [";
        for (size_t i = 0; i < fixed.size(); ++i) {
            if (i) js += ", ";
            js += std::to_string(fixed[i]);
        }
        js += "],\n  \"top2_margin_u6\": [";
        for (size_t i = 0; i < margins.size(); ++i) {
            if (i) js += ", ";
            js += std::to_string(margins[i]);
        }
        js += "],\n  \"input_tokens\": " + std::to_string(items.size());
        js += ",\n  \"audio_tokens\": " + std::to_string(official_audio_tokens);
        {
            char buf[96];
            snprintf(buf, sizeof(buf), ",\n  \"forward_seconds\": %.6f", forward_s);
            js += buf;
        }
        js += "\n}\n";

        if (opt.output == "-") {
            fputs(js.c_str(), stdout);
        } else {
            std::ofstream f(opt.output);
            if (!f) {
                fprintf(stderr, "error: cannot write %s\n", opt.output.c_str());
                return 1;
            }
            f << js;
        }

        mtmd_bitmap_free(bm.bitmap);
        mtmd_free(mctx);
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
