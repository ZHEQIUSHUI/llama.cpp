#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "forced-aligner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

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

namespace mtmd_forced_aligner {

result run(llama_model * model, llama_context * ctx, mtmd_context * mctx,
           const void * audio, size_t audio_len, const options & opt) {
    auto fail = [](int code, const std::string & message) {
        result r;
        r.code = code;
        r.error = message;
        return r;
    };
    try {
        if (!model || !ctx || !mctx || !audio || audio_len == 0) {
            return fail(400, "audio input is required");
        }
        if (opt.transcript.empty()) {
            return fail(400, "transcript is required");
        }
        if (opt.segment_ms <= 0) {
            return fail(400, "segment_ms must be positive");
        }
        std::string language = opt.language;
        std::transform(language.begin(), language.end(), language.begin(),
                       [](unsigned char c) { return (char) std::tolower(c); });
        if (language == "japanese" || language == "korean") {
            return fail(400, "Japanese and Korean are not supported: their official word segmentation requires "
                              "the nagisa and soynlp dictionaries, which are not part of this self-contained binary");
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const int n_class = llama_model_n_embd_out(model);
        const int n_embd = llama_model_n_embd_inp(model);
        if (n_class <= 0 || n_class > n_vocab) {
            return fail(400, "the loaded model has no valid classifier output width");
        }
        llama_set_sampler(ctx, 0, nullptr);

        mtmd_helper_bitmap_wrapper wrapper = mtmd_helper_bitmap_init_from_buf(
                mctx, (const unsigned char *) audio, audio_len, false, mtmd_helper_init_opt_default());
        std::unique_ptr<mtmd_bitmap, decltype(&mtmd_bitmap_free)> bitmap(wrapper.bitmap, mtmd_bitmap_free);
        if (!bitmap || !mtmd_bitmap_is_audio(bitmap.get())) {
            return fail(400, "failed to decode the input audio");
        }
        const int64_t n_samples = (int64_t) mtmd_bitmap_get_nx(bitmap.get());

        std::vector<std::string> word_list = split_words(opt.transcript);
        if (word_list.empty()) {
            return fail(400, "transcript has no alignable words");
        }
        const std::string suffix = timestamp_prompt(word_list);

        mtmd_input_text text_empty = { "", 0, false, true };
        mtmd_input_text text_suffix = { suffix.data(), suffix.size(), false, true };
        mtmd_input_part parts[3] = {
            { &text_empty, nullptr },
            { nullptr, bitmap.get() },
            { &text_suffix, nullptr },
        };
        const mtmd_input_part * part_ptrs[3] = { &parts[0], &parts[1], &parts[2] };
        std::unique_ptr<mtmd_input_chunks, decltype(&mtmd_input_chunks_free)> chunks(
                mtmd_input_chunks_init(), mtmd_input_chunks_free);
        if (mtmd_tokenize_from_parts(mctx, chunks.get(), part_ptrs, 3, false) != 0) {
            return fail(400, "failed to tokenize the audio and transcript");
        }

        std::vector<llama_token> ts;
        ts.resize(32);
        const int n_ts = llama_tokenize(vocab, "<timestamp>", 11, ts.data(), (int) ts.size(), false, true);
        if (n_ts != 1) {
            return fail(500, "the tokenizer does not map <timestamp> to one token");
        }
        const size_t ts_id = (size_t) ts[0];

        const int64_t total_frames = n_samples / 160;
        int64_t window_off = 0;
        size_t audio_chunk_index = 0;
        int64_t audio_tokens = 0;
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
                    return fail(400, "more audio chunks than mel windows");
                }
                const int64_t want = feat_output_length(win_eff);
                const size_t got = mtmd_input_chunk_get_n_tokens(chunk);
                if ((int64_t) got < want) {
                    return fail(400, "the audio encoder produced fewer rows than the official window");
                }
                if (mtmd_encode_chunk(mctx, chunk) != 0) {
                    return fail(500, "audio encoder failed");
                }
                const float * embd = mtmd_get_output_embd(mctx);
                if (!embd) {
                    return fail(500, "audio encoder returned no embeddings");
                }
                for (int64_t r = 0; r < want; ++r) {
                    seq_item item;
                    item.embd = true;
                    item.audio_row = audio_rows.size();
                    audio_rows.emplace_back(embd + (size_t) r * n_embd, embd + (size_t) (r + 1) * n_embd);
                    items.push_back(item);
                }
                audio_tokens += want;
                window_off += win_eff;
                ++audio_chunk_index;
            }
        }
        if (window_off < total_frames) {
            return fail(400, "the audio chunks do not cover the complete input");
        }
        if (timestamp_items.size() != word_list.size() * 2) {
            return fail(400, "timestamp token count does not match the transcript");
        }
        const int64_t n_ctx = llama_n_ctx(ctx);
        const int64_t n_batch = llama_n_batch(ctx);
        if ((int64_t) items.size() > n_batch) {
            return fail(400, "prompt has " + std::to_string(items.size()) + " tokens, the server batch limit is " +
                            std::to_string(n_batch));
        }
        if ((int64_t) items.size() > n_ctx) {
            return fail(400, "prompt has " + std::to_string(items.size()) + " tokens, the context limit is " +
                            std::to_string(n_ctx));
        }

        llama_synchronize(ctx);
        llama_memory_clear(llama_get_memory(ctx), true);

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
                return fail(500, "failed to build the model input batch");
            }
            llama_pos p[4] = { (llama_pos) i, (llama_pos) i, (llama_pos) i, (llama_pos) i };
            if (!llama_batch_ext_set_pos(batch.get(), idx, p)) {
                return fail(500, "failed to set an input position");
            }
        }
        for (size_t idx : timestamp_items) {
            llama_batch_ext_set_output_logits(batch.get(), (int32_t) idx, true);
        }

        const auto t_forward = std::chrono::steady_clock::now();
        if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get()) != 0) {
            llama_memory_clear(llama_get_memory(ctx), true);
            return fail(500, "the model forward pass failed");
        }
        llama_synchronize(ctx);
        const double forward_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_forward).count();

        std::vector<int64_t> raw_ms;
        std::vector<int> argmax_ids;
        std::vector<int> margins;
        raw_ms.reserve(timestamp_items.size());
        argmax_ids.reserve(timestamp_items.size());
        margins.reserve(timestamp_items.size());
        for (size_t idx : timestamp_items) {
            const float * row = llama_get_logits_ith(ctx, (int32_t) idx);
            if (!row) {
                llama_memory_clear(llama_get_memory(ctx), true);
                return fail(500, "the model returned no logits for a timestamp position");
            }
            int margin = 0;
            const int id = argmax_class(row, n_vocab, n_class, &margin);
            if (id < 0 || id >= n_class) {
                llama_memory_clear(llama_get_memory(ctx), true);
                return fail(500, "the classifier returned an invalid class");
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
        js += "\n  ],\n  \"segments\": [";
        if (!spans.empty()) {
            char buf[160];
            snprintf(buf, sizeof(buf),
                     "\n    {\"id\": 0, \"start\": %.3f, \"end\": %.3f, \"text\": ", spans.front().start,
                     spans.back().end);
            js += buf;
            json_string(js, opt.transcript);
            js += ", \"words\": []}";
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
        js += ",\n  \"audio_tokens\": " + std::to_string(audio_tokens);
        {
            char buf[96];
            snprintf(buf, sizeof(buf), ",\n  \"forward_seconds\": %.6f", forward_s);
            js += buf;
        }
        js += "\n}\n";

        llama_memory_clear(llama_get_memory(ctx), true);

        result r;
        r.ok = true;
        r.code = 200;
        r.json = std::move(js);
        r.input_tokens = (int) items.size();
        r.audio_tokens = (int) audio_tokens;
        r.forward_seconds = forward_s;
        return r;
    } catch (const std::exception & e) {
        return fail(500, std::string("alignment failed: ") + e.what());
    }
}

} // namespace mtmd_forced_aligner
