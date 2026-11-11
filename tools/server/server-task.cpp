#include "server-task.h"

#include "build-info.h"
#include "server-chat.h"
#include "chat.h"
#include "common.h"
#include "json-schema-to-grammar.h"
#include "llama.h"
#include "sampling.h"
#include "speculative.h"
#include "server-common.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

//
// task_params
//

json task_params::format_logit_bias(const std::vector<llama_logit_bias> & logit_bias) const {
    json data = json::array();
    for (const auto & lb : logit_bias) {
        data.push_back(json{
            {"bias", lb.bias},
            {"token", lb.token},
        });
    }
    return data;
}

json task_params::to_json(bool only_metrics) const {
    std::vector<std::string> samplers;
    samplers.reserve(sampling.samplers.size());
    for (const auto & sampler : sampling.samplers) {
        samplers.emplace_back(common_sampler_type_to_str(sampler));
    }

    json lora = json::array();
    for (auto & it : this->lora) {
        lora.push_back({{"id", it.first}, {"scale", it.second}});
    }

    if (only_metrics) {
        return json {
            {"seed",                      sampling.seed},
            {"temperature",               sampling.temp},
            {"dynatemp_range",            sampling.dynatemp_range},
            {"dynatemp_exponent",         sampling.dynatemp_exponent},
            {"top_k",                     sampling.top_k},
            {"top_p",                     sampling.top_p},
            {"min_p",                     sampling.min_p},
            {"top_n_sigma",               sampling.top_n_sigma},
            {"xtc_probability",           sampling.xtc_probability},
            {"xtc_threshold",             sampling.xtc_threshold},
            {"typical_p",                 sampling.typ_p},
            {"repeat_last_n",             sampling.penalty_last_n},
            {"repeat_penalty",            sampling.penalty_repeat},
            {"presence_penalty",          sampling.penalty_present},
            {"frequency_penalty",         sampling.penalty_freq},
            {"dry_multiplier",            sampling.dry_multiplier},
            {"dry_base",                  sampling.dry_base},
            {"dry_allowed_length",        sampling.dry_allowed_length},
            {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
            {"mirostat",                  sampling.mirostat},
            {"mirostat_tau",              sampling.mirostat_tau},
            {"mirostat_eta",              sampling.mirostat_eta},
            {"adaptive_target",           sampling.adaptive_target},
            {"adaptive_decay",            sampling.adaptive_decay},
            {"max_tokens",                n_predict},
            {"n_predict",                 n_predict}, // TODO: deduplicate?
            {"n_keep",                    n_keep},
            {"n_discard",                 n_discard},
            {"ignore_eos",                sampling.ignore_eos},
            {"stream",                    stream},
            {"n_probs",                   sampling.n_probs},
            {"min_keep",                  sampling.min_keep},
            {"chat_format",               common_chat_format_name(chat_format)},
            {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
            {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
            {"generation_prompt",         sampling.generation_prompt},
            {"samplers",                  samplers},
            {"speculative.types",         common_speculative_type_name_str(speculative.types)},
            {"timings_per_token",         timings_per_token},
            {"post_sampling_probs",       post_sampling_probs},
            {"backend_sampling",          sampling.backend_sampling},
            {"lora",                      lora},
        };
    }

    auto grammar_triggers = json::array();
    for (const auto & trigger : sampling.grammar_triggers) {
        server_grammar_trigger ct(trigger);
        grammar_triggers.push_back(ct.to_json());
    }

    return json {
        {"seed",                      sampling.seed},
        {"temperature",               sampling.temp},
        {"dynatemp_range",            sampling.dynatemp_range},
        {"dynatemp_exponent",         sampling.dynatemp_exponent},
        {"top_k",                     sampling.top_k},
        {"top_p",                     sampling.top_p},
        {"min_p",                     sampling.min_p},
        {"top_n_sigma",               sampling.top_n_sigma},
        {"xtc_probability",           sampling.xtc_probability},
        {"xtc_threshold",             sampling.xtc_threshold},
        {"typical_p",                 sampling.typ_p},
        {"repeat_last_n",             sampling.penalty_last_n},
        {"repeat_penalty",            sampling.penalty_repeat},
        {"presence_penalty",          sampling.penalty_present},
        {"frequency_penalty",         sampling.penalty_freq},
        {"dry_multiplier",            sampling.dry_multiplier},
        {"dry_base",                  sampling.dry_base},
        {"dry_allowed_length",        sampling.dry_allowed_length},
        {"dry_penalty_last_n",        sampling.dry_penalty_last_n},
        {"dry_sequence_breakers",     sampling.dry_sequence_breakers},
        {"mirostat",                  sampling.mirostat},
        {"mirostat_tau",              sampling.mirostat_tau},
        {"mirostat_eta",              sampling.mirostat_eta},
        {"adaptive_target",           sampling.adaptive_target},
        {"adaptive_decay",            sampling.adaptive_decay},
        {"stop",                      antiprompt},
        {"max_tokens",                n_predict},
        {"n_predict",                 n_predict}, // TODO: deduplicate?
        {"n_keep",                    n_keep},
        {"n_discard",                 n_discard},
        {"ignore_eos",                sampling.ignore_eos},
        {"stream",                    stream},
        {"logit_bias",                format_logit_bias(sampling.logit_bias)},
        {"n_probs",                   sampling.n_probs},
        {"min_keep",                  sampling.min_keep},
        {"grammar",                   common_grammar_value(sampling.grammar)},
        {"grammar_lazy",              sampling.grammar_lazy},
        {"grammar_triggers",          grammar_triggers},
        {"preserved_tokens",          sampling.preserved_tokens},
        {"chat_format",               common_chat_format_name(chat_format)},
        {"reasoning_format",          common_reasoning_format_name(reasoning_format)},
        {"reasoning_in_content",      stream && reasoning_format == COMMON_REASONING_FORMAT_DEEPSEEK_LEGACY},
        {"generation_prompt",         sampling.generation_prompt},
        {"samplers",                  samplers},
        {"speculative.types",         common_speculative_type_name_str(speculative.types)},
        {"timings_per_token",         timings_per_token},
        {"post_sampling_probs",       post_sampling_probs},
        {"backend_sampling",          sampling.backend_sampling},
        {"lora",                      lora},
    };
}

//
// task_result_state
//
task_result_state::task_result_state(common_chat_session session)
    : chat_session(std::move(session))
    , chat_msg(chat_session.msg())
    , oai_resp_id("resp_" + random_string())
    , oai_resp_reasoning_id("rs_" + random_string())
    , oai_resp_message_id("msg_" + random_string()) {
}

common_chat_msg task_result_state::update_chat_msg(
        const common_chat_input & added,
        bool is_partial,
        std::vector<common_chat_msg_diff> & diffs,
        bool filter_tool_calls) {
    auto msg_prv_copy = chat_msg;
    auto new_msg = is_partial ? chat_session.feed(added) : chat_session.finish(added);
    if (!new_msg.empty()) {
        new_msg.set_tool_call_ids(generated_tool_call_ids, gen_tool_call_id);
        chat_msg = new_msg;
        auto all_diffs = common_chat_msg_diff::compute_diffs(msg_prv_copy, chat_msg);

        if (!filter_tool_calls) {
            diffs = std::move(all_diffs);
        } else {
            for (auto & d : all_diffs) {
                // If this is a new type of delta, flush all currently pending tool call names
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (sent_tool_call_names.count(i) || chat_msg.tool_calls[i].name.empty()) {
                        continue;
                    }
                    if (d.tool_call_index != i || !d.tool_call_delta.arguments.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }

                if (d.tool_call_index == std::string::npos) {
                    diffs.push_back(std::move(d));
                } else {
                    size_t i = d.tool_call_index;
                    if (sent_tool_call_names.count(i)) {
                        if (!d.tool_call_delta.arguments.empty()) {
                            d.tool_call_delta.name = "";
                            d.tool_call_delta.id   = "";
                            diffs.push_back(std::move(d));
                        }
                    } else {
                        // Not sent yet.
                        if (!d.tool_call_delta.arguments.empty() || !is_partial) {
                            d.tool_call_delta.name = chat_msg.tool_calls[i].name;
                            d.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                            diffs.push_back(std::move(d));
                            sent_tool_call_names.insert(i);
                        } else {
                            // Suppress
                        }
                    }
                }
            }
            // Final check at EOF
            if (!is_partial) {
                for (size_t i = 0; i < chat_msg.tool_calls.size(); ++i) {
                    if (!sent_tool_call_names.count(i) && !chat_msg.tool_calls[i].name.empty()) {
                        common_chat_msg_diff header;
                        header.tool_call_index      = i;
                        header.tool_call_delta.id   = chat_msg.tool_calls[i].id;
                        header.tool_call_delta.name = chat_msg.tool_calls[i].name;
                        diffs.push_back(std::move(header));
                        sent_tool_call_names.insert(i);
                    }
                }
            }
        }
    }
    return chat_msg;
}

//
// result_prompt_progress
//
json result_prompt_progress::to_json() const {
    return json {
        {"total",     total},
        {"cache",     cache},
        {"processed", processed},
        {"time_ms",   time_ms},
    };
}

static inline std::string stop_type_to_str(stop_type type) {
    switch (type) {
        case STOP_TYPE_EOS:   return "eos";
        case STOP_TYPE_WORD:  return "word";
        case STOP_TYPE_LIMIT: return "limit";
        default:              return "none";
    }
}

//
// completion_token_output
//

json completion_token_output::to_json(bool post_sampling_probs) const {
    json probs_for_token = json::array();
    for (const auto & p : probs) {
        std::string txt(p.txt);
        txt.resize(validate_utf8(txt));
        probs_for_token.push_back(json {
            {"id",      p.tok},
            {"token",   txt},
            {"bytes",   str_to_bytes(p.txt)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
        });
    }
    return probs_for_token;
}

json completion_token_output::probs_vector_to_json(const std::vector<completion_token_output> & probs, bool post_sampling_probs) {
    json out = json::array();
    for (const auto & p : probs) {
        std::string txt(p.text_to_send);
        txt.resize(validate_utf8(txt));
        out.push_back(json {
            {"id",           p.tok},
            {"token",        txt},
            {"bytes",        str_to_bytes(p.text_to_send)},
            {
                post_sampling_probs ? "prob" : "logprob",
                post_sampling_probs ? p.prob : logarithm(p.prob)
            },
            {
                post_sampling_probs ? "top_probs" : "top_logprobs",
                p.to_json(post_sampling_probs)
            },
        });
    }
    return out;
}

float completion_token_output::logarithm(float x) {
    // the JSON library converts -inf to null, so we need to prevent that
    return x == 0.0f ? std::numeric_limits<float>::lowest() : std::log(x);
}

std::vector<unsigned char> completion_token_output::str_to_bytes(const std::string & str) {
    std::vector<unsigned char> bytes;
    for (unsigned char c : str) {
        bytes.push_back(c);
    }
    return bytes;
}

//
// server_task_result_cmpl_final
//
json server_task_result_cmpl_final::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return stream ? to_json_oaicompat_chat_stream() : to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return stream ? to_json_oaicompat_resp_stream() : to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return stream ? to_json_anthropic_stream() : to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_final::to_json_non_oaicompat() {
    json res = json {
        {"index",               index},
        {"content",             content.text},
        {"tokens",              tokens},
        {"id_slot",             id_slot},
        {"stop",                true},
        {"model",               oaicompat_model},
        {"tokens_predicted",    n_decoded},
        {"tokens_evaluated",    n_prompt_tokens},
        {"generation_settings", generation_params.to_json()},
        {"prompt",              prompt},
        {"has_new_line",        has_new_line},
        {"truncated",           truncated},
        {"stop_type",           stop_type_to_str(stop)},
        {"stopping_word",       stopping_word},
        {"tokens_cached",       n_tokens_cached},
        {"timings",             stats.to_json()},
    };
    if (!stream && !probs_output.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs);
    }
    return response_fields.empty() ? res : json_get_nested_values(response_fields, res);
}

json server_task_result_cmpl_final::usage_json_oaicompat() {
    return json {
        {"completion_tokens", n_decoded},
        {"prompt_tokens",     n_prompt_tokens},
        {"total_tokens",      n_decoded + n_prompt_tokens},
        {"prompt_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
    };
}

json server_task_result_cmpl_final::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (!stream && probs_output.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }
    json finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = "stop";
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", finish_reason},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat() {
    std::string finish_reason = "length";
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json choice {
        {"finish_reason", finish_reason},
        {"index", index},
        {"message", msg.to_json_oaicompat()},
    };

    if (!stream && probs_output.size() > 0) {
        choice["logprobs"] = json{
            {"content", completion_token_output::probs_vector_to_json(probs_output, post_sampling_probs)},
        };
    }

    std::time_t t = std::time(0);

    json res = json {
        {"choices",            json::array({choice})},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion"},
        {"usage",              usage_json_oaicompat()},
        {"id", oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_chat_stream() {
    std::time_t t = std::time(0);
    std::string finish_reason = "length";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        finish_reason = oaicompat_msg.tool_calls.empty() ? "stop" : "tool_calls";
    }

    json deltas = json::array();
    for (const auto & diff : oaicompat_msg_diffs) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", server_chat_msg_diff_to_json_oaicompat(diff)},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    }

    deltas.push_back({
        {"choices", json::array({
            json {
                {"finish_reason", finish_reason},
                {"index", index},
                {"delta", json::object()},
            },
        })},
        {"created",            t},
        {"id",                 oaicompat_cmpl_id},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "chat.completion.chunk"},
    });

    if (include_usage) {
        // OpenAI API spec for chat.completion.chunks specifies an empty `choices` array for the last chunk when including usage
        // https://platform.openai.com/docs/api-reference/chat_streaming/streaming#chat_streaming/streaming-choices
        deltas.push_back({
            {"choices", json::array()},
            {"created",            t},
            {"id",                 oaicompat_cmpl_id},
            {"model",              oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object",             "chat.completion.chunk"},
            {"usage",              usage_json_oaicompat()},
        });
    }

    if (stats.is_set()) {
        deltas.back()["timings"] = stats.to_json();
    }

    // extra fields for debugging purposes
    if (verbose && !deltas.empty()) {
        deltas.front()["__verbose"] = to_json_non_oaicompat();
    }

    return deltas;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp() {
    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    std::vector<json> output;

    if (msg.reasoning_content != "") {
        output.push_back(json {
            {"id",      "rs_" + random_string()},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
            {"status",            "completed"},
        });
    }

    if (msg.content != "") {
        output.push_back(json {
            {"content", json::array({ json {
                {"type",        "output_text"},
                {"annotations", json::array()},
                {"logprobs",    json::array()},
                {"text",        msg.content},
            }})},
            {"id",     "msg_" + random_string()},
            {"role",   msg.role},
            {"status", "completed"},
            {"type",   "message"},
        });
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        output.push_back(json {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name},
        });
    }

    std::time_t t = std::time(0);
    json res = {
        {"completed_at", t},
        {"created_at",   t},
        {"id",           oai_resp_id},
        {"model",        oaicompat_model},
        {"object",       "response"},
        {"output",       output},
        {"status",       "completed"},
        {"usage",        json {
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };

    return res;
}

json server_task_result_cmpl_final::to_json_oaicompat_resp_stream() {
    std::vector<json> server_sent_events;
    std::vector<json> output;

    if (oaicompat_msg.reasoning_content != "") {
        const json output_item = json {
            {"id",      oai_resp_reasoning_id},
            {"summary", json::array()},
            {"type",    "reasoning"},
            {"content", json::array({ json {
                {"text", oaicompat_msg.reasoning_content},
                {"type", "reasoning_text"},
            }})},
            {"encrypted_content", ""},
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    if (oaicompat_msg.content != "") {
        server_sent_events.push_back(json {
            {"event", "response.output_text.done"},
            {"data", json {
                {"type",    "response.output_text.done"},
                {"item_id", oai_resp_message_id},
                {"text",    oaicompat_msg.content}
            }}
        });

        const json content_part = {
            {"type",        "output_text"},
            {"annotations", json::array()},
            {"logprobs",    json::array()},
            {"text",        oaicompat_msg.content}
        };

        server_sent_events.push_back(json {
            {"event", "response.content_part.done"},
            {"data", json {
                {"type",    "response.content_part.done"},
                {"item_id", oai_resp_message_id},
                {"part",    content_part}
            }}
        });
        const json output_item = {
            {"type",    "message"},
            {"status",  "completed"},
            {"id",      oai_resp_message_id},
            {"content", json::array({content_part})},
            {"role",    "assistant"}
        };

        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    for (const common_chat_tool_call & tool_call : oaicompat_msg.tool_calls) {
        const json output_item = {
            {"id",        "fc_" + tool_call.id},
            {"type",      "function_call"},
            {"status",    "completed"},
            {"arguments", tool_call.arguments},
            {"call_id",   "call_" + tool_call.id},
            {"name",      tool_call.name}
        };
        server_sent_events.push_back(json {
            {"event", "response.output_item.done"},
            {"data", json {
                {"type", "response.output_item.done"},
                {"item", output_item}
            }}
        });
        output.push_back(output_item);
    }

    std::time_t t = std::time(0);
    server_sent_events.push_back(json {
        {"event", "response.completed"},
        {"data", json {
            {"type", "response.completed"},
            {"response", json {
                {"id",         oai_resp_id},
                {"object",     "response"},
                {"created_at", t},
                {"status",     "completed"},
                {"model",      oaicompat_model},
                {"output",     output},
                {"usage",      json {
                    {"input_tokens",  n_prompt_tokens},
                    {"output_tokens", n_decoded},
                    {"total_tokens",  n_decoded + n_prompt_tokens},
                    {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
                }}
            }},
        }}
    });

    if (stats.is_set()) {
        server_sent_events.back().at("data")["timings"] = stats.to_json();
    }

    return server_sent_events;
}

json server_task_result_cmpl_final::to_json_oaicompat_asr() {
    json event = json {
        {"type",  "transcript.text.done"},
        {"text",  oaicompat_msg.content},
        {"usage", json {
            {"type",         "tokens"},
            {"input_tokens",  n_prompt_tokens},
            {"output_tokens", n_decoded},
            {"total_tokens",  n_decoded + n_prompt_tokens},
            {"input_tokens_details", json { {"cached_tokens", n_prompt_tokens_cache} }},
        }},
    };
    return event;
}

json server_task_result_cmpl_final::to_json_anthropic() {
    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    json content_blocks = json::array();

    common_chat_msg msg;
    if (!oaicompat_msg.empty()) {
        msg = oaicompat_msg;
    } else {
        msg.role = "assistant";
        msg.content = content.text;
    }

    // thinking block comes first (Anthropic extended thinking format)
    if (!msg.reasoning_content.empty()) {
        content_blocks.push_back({
            {"type", "thinking"},
            {"thinking", msg.reasoning_content},
            {"signature", ""}  // empty signature for local models (no cryptographic verification)
        });
    }

    if (!msg.content.empty()) {
        content_blocks.push_back({
            {"type", "text"},
            {"text", msg.content}
        });
    }

    for (const auto & tool_call : msg.tool_calls) {
        json tool_use_block = {
            {"type", "tool_use"},
            {"id", tool_call.id},
            {"name", tool_call.name}
        };

        try {
            tool_use_block["input"] = json::parse(tool_call.arguments);
        } catch (const std::exception &) {
            tool_use_block["input"] = json::object();
        }

        content_blocks.push_back(tool_use_block);
    }

    json res = {
        {"id", oaicompat_cmpl_id},
        {"type", "message"},
        {"role", "assistant"},
        {"content", content_blocks},
        {"model", oaicompat_model},
        {"stop_reason", stop_reason},
        {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)},
        {"usage", {
            {"cache_read_input_tokens", n_prompt_tokens_cache},
            {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
            {"output_tokens", n_decoded}
        }}
    };

    return res;
}

json server_task_result_cmpl_final::to_json_anthropic_stream() {
    json events = json::array();

    std::string stop_reason = "max_tokens";
    if (stop == STOP_TYPE_WORD || stop == STOP_TYPE_EOS) {
        stop_reason = oaicompat_msg.tool_calls.empty() ? "end_turn" : "tool_use";
    }

    bool has_thinking = !oaicompat_msg.reasoning_content.empty();
    bool has_text     = !oaicompat_msg.content.empty();
    size_t num_tool_calls = oaicompat_msg.tool_calls.size();

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    size_t text_block_index     = has_thinking ? 1 : 0;

    bool thinking_block_started = false;
    bool text_block_started     = false;
    std::unordered_set<size_t> tool_calls_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_block_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + diff.tool_call_index;

            if (tool_calls_started.find(diff.tool_call_index) == tool_calls_started.end()) {
                const auto & full_tool_call = oaicompat_msg.tool_calls[diff.tool_call_index];

                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", full_tool_call.id},
                            {"name", full_tool_call.name}
                        }}
                    }}
                });
                tool_calls_started.insert(diff.tool_call_index);
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    // close content blocks in order
    if (has_thinking) {
        // Anthropic API requires a signature_delta before closing thinking blocks
        // We use an empty signature since we can't generate a cryptographic signature for local models
        events.push_back({
            {"event", "content_block_delta"},
            {"data", {
                {"type", "content_block_delta"},
                {"index", thinking_block_index},
                {"delta", {
                    {"type", "signature_delta"},
                    {"signature", ""}
                }}
            }}
        });
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", thinking_block_index}
            }}
        });
    }

    if (has_text) {
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", text_block_index}
            }}
        });
    }

    for (size_t i = 0; i < num_tool_calls; i++) {
        size_t content_block_index = (has_thinking ? 1 : 0) + (has_text ? 1 : 0) + i;
        events.push_back({
            {"event", "content_block_stop"},
            {"data", {
                {"type", "content_block_stop"},
                {"index", content_block_index}
            }}
        });
    }

    events.push_back({
        {"event", "message_delta"},
        {"data", {
            {"type", "message_delta"},
            {"delta", {
                {"stop_reason", stop_reason},
                {"stop_sequence", stopping_word.empty() ? nullptr : json(stopping_word)}
            }},
            {"usage", {
                {"output_tokens", n_decoded}
            }}
        }}
    });

    events.push_back({
        {"event", "message_stop"},
        {"data", {
            {"type", "message_stop"}
        }}
    });

    return events;
}

//
// server_task_result_cmpl_partial
//
void server_task_result_cmpl_partial::update(task_result_state & state) {
    is_updated = true;
    if (is_begin) {
        return; // begin marker only flushes headers, skip parsing
    }
    state.update_chat_msg(content, true, oaicompat_msg_diffs);

    // Copy current state for use in to_json_*() (reflects state BEFORE this chunk)
    thinking_block_started = state.thinking_block_started;
    text_block_started     = state.text_block_started;

    oai_resp_created       = state.oai_resp_created;
    oai_resp_id            = state.oai_resp_id;
    oai_resp_reasoning_id  = state.oai_resp_reasoning_id;
    oai_resp_message_id    = state.oai_resp_message_id;
    oai_resp_fc_id         = state.oai_resp_fc_id;

    // track if the accumulated message has any reasoning content
    anthropic_has_reasoning = !state.chat_msg.reasoning_content.empty();

    if (res_type == TASK_RESPONSE_TYPE_OAI_RESP && !state.oai_resp_created && (is_progress || n_decoded == 1)) {
        state.oai_resp_created = true;
    }

    // Pre-compute state updates based on diffs (for next chunk)
    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty() && !state.thinking_block_started) {
            state.thinking_block_started = true;
        }
        if (!diff.content_delta.empty() && !state.text_block_started) {
            state.text_block_started = true;
        }
        if (!diff.tool_call_delta.name.empty()) {
            state.oai_resp_fc_id = diff.tool_call_delta.id;
        }
    }
}

json server_task_result_cmpl_partial::to_json() {
    GGML_ASSERT(is_updated && "update() must be called before to_json()");
    if (is_begin) {
        return nullptr; // simply signal to HTTP handler to send the headers and status code
    }
    switch (res_type) {
        case TASK_RESPONSE_TYPE_NONE:
            return to_json_non_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CMPL:
            return to_json_oaicompat();
        case TASK_RESPONSE_TYPE_OAI_CHAT:
            return to_json_oaicompat_chat();
        case TASK_RESPONSE_TYPE_OAI_RESP:
            return to_json_oaicompat_resp();
        case TASK_RESPONSE_TYPE_OAI_ASR:
            return to_json_oaicompat_asr();
        case TASK_RESPONSE_TYPE_ANTHROPIC:
            return to_json_anthropic();
        default:
            GGML_ASSERT(false && "Invalid task_response_type");
    }
}

json server_task_result_cmpl_partial::to_json_non_oaicompat() {
    // non-OAI-compat JSON
    json res = json {
        {"index",            index},
        {"content",          content.text},
        {"tokens",           tokens},
        {"stop",             false},
        {"id_slot",          id_slot},
        {"tokens_predicted", n_decoded},
        {"tokens_evaluated", n_prompt_tokens},
    };
    // populate the timings object when needed (usually for the last response or with timings_per_token enabled)
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }
    if (!prob_output.probs.empty()) {
        res["completion_probabilities"] = completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs);
    }
    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat() {
    std::time_t t = std::time(0);
    json logprobs = json(nullptr); // OAI default to null
    if (prob_output.probs.size() > 0) {
        logprobs = json{
            {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
        };
    }
    json res = json {
        {"choices",            json::array({
            json{
                {"text",          content.text},
                {"index",         index},
                {"logprobs",      logprobs},
                {"finish_reason", nullptr},
            }
        })},
        {"created",            t},
        {"model",              oaicompat_model},
        {"system_fingerprint", std::string(llama_build_info())},
        {"object",             "text_completion"},
        {"id",                 oaicompat_cmpl_id}
    };

    // extra fields for debugging purposes
    if (verbose) {
        res["__verbose"] = to_json_non_oaicompat();
    }
    if (stats.is_set()) {
        res["timings"] = stats.to_json();
    }
    if (is_progress) {
        res["prompt_progress"] = progress.to_json();
    }

    return res;
}

json server_task_result_cmpl_partial::to_json_oaicompat_chat() {
    bool first = n_decoded == 1;
    std::time_t t = std::time(0);
    json choices;

    std::vector<json> deltas;
    auto add_delta = [&](const json & delta) {
        deltas.push_back({
            {"choices", json::array({
                json {
                    {"finish_reason", nullptr},
                    {"index", index},
                    {"delta", delta},
                },
            })},
            {"created", t},
            {"id", oaicompat_cmpl_id},
            {"model", oaicompat_model},
            {"system_fingerprint", std::string(llama_build_info())},
            {"object", "chat.completion.chunk"},
        });
    };
    // We have to send an initial update to conform to openai behavior
    if (first || is_progress) {
        add_delta({
            {"role", "assistant"},
            {"content", nullptr},
        });
    }

    for (const auto & diff : oaicompat_msg_diffs) {
        add_delta(server_chat_msg_diff_to_json_oaicompat(diff));
    }

    if (!deltas.empty()) {
        auto & last_json = deltas[deltas.size() - 1];
        GGML_ASSERT(last_json.at("choices").size() >= 1);

        if (prob_output.probs.size() > 0) {
            last_json.at("choices").at(0)["logprobs"] = json {
                {"content", completion_token_output::probs_vector_to_json({prob_output}, post_sampling_probs)},
            };
        }

        if (stats.is_set()) {
            last_json["timings"] = stats.to_json();
        }
        if (is_progress) {
            last_json["prompt_progress"] = progress.to_json();
        }
    }

    return deltas;
}

json server_task_result_cmpl_partial::to_json_oaicompat_resp() {
    std::vector<json> events;

    if (!oai_resp_created) {
        events.push_back(json {
            {"event", "response.created"},
            {"data", json {
                {"type", "response.created"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    } else if (is_progress) {
        events.push_back(json {
            {"event", "response.in_progress"},
            {"data", json {
                {"type", "response.in_progress"},
                {"response", json {
                    {"id",     oai_resp_id},
                    {"object", "response"},
                    {"status", "in_progress"},
                }},
            }},
        });
    }

    for (const common_chat_msg_diff & diff : oaicompat_msg_diffs) {
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"id",                oai_resp_reasoning_id},
                            {"summary",           json::array()},
                            {"type",              "reasoning"},
                            {"content",           json::array()},
                            {"encrypted_content", ""},
                            {"status",            "in_progress"},
                        }},
                    }},
                });
                thinking_block_started = true;
            }
            events.push_back(json {
                {"event", "response.reasoning_text.delta"},
                {"data", json {
                    {"type",    "response.reasoning_text.delta"},
                    {"delta",   diff.reasoning_content_delta},
                    {"item_id", oai_resp_reasoning_id},
                }},
            });
        }

        if (!diff.content_delta.empty()) {
            if (!text_block_started) {
                events.push_back(json {
                    {"event", "response.output_item.added"},
                    {"data", json {
                        {"type", "response.output_item.added"},
                        {"item", json {
                            {"content", json::array()},
                            {"id",      oai_resp_message_id},
                            {"role",    "assistant"},
                            {"status",  "in_progress"},
                            {"type",    "message"},
                        }},
                    }},
                });
                events.push_back(json {
                    {"event", "response.content_part.added"},
                    {"data", json {
                        {"type",    "response.content_part.added"},
                        {"item_id", oai_resp_message_id},
                        {"part", json {
                            {"type", "output_text"},
                            {"text", ""},
                        }},
                    }},
                });
                text_block_started = true;
            }
            events.push_back(json {
                {"event", "response.output_text.delta"},
                {"data", json {
                    {"type",    "response.output_text.delta"},
                    {"item_id", oai_resp_message_id},
                    {"delta",   diff.content_delta},
                }},
            });
        }

        if (!diff.tool_call_delta.name.empty()) {
            events.push_back(json {
                {"event", "response.output_item.added"},
                {"data", json {
                    {"type",  "response.output_item.added"},
                    {"item", json {
                        {"id",        "fc_" + diff.tool_call_delta.id},
                        {"arguments", ""},
                        {"call_id",   "call_" + diff.tool_call_delta.id},
                        {"name",      diff.tool_call_delta.name},
                        {"type",      "function_call"},
                        {"status",    "in_progress"},
                    }},
                }},
            });
            oai_resp_fc_id = diff.tool_call_delta.id;
        }

        if (!diff.tool_call_delta.arguments.empty()) {
            events.push_back(json {
                {"event", "response.function_call_arguments.delta"},
                {"data", json {
                    {"type",    "response.function_call_arguments.delta"},
                    {"delta",   diff.tool_call_delta.arguments},
                    {"item_id", "fc_" + oai_resp_fc_id},
                }},
            });
        }
    }

    if (!events.empty()) {
        json & data = events.back().at("data");
        if (stats.is_set()) {
            data["timings"] = stats.to_json();
        }
        if (is_progress) {
            data["prompt_progress"] = progress.to_json();
        }
    }

    return events;
}

json server_task_result_cmpl_partial::to_json_oaicompat_asr() {
    json event = json {
        {"type", "transcript.text.delta"},
        {"delta", content.text},
    };
    return event;
}

json server_task_result_cmpl_partial::to_json_anthropic() {
    json events = json::array();
    bool first = (n_decoded == 1);
    // use member variables to track block state across streaming calls
    // (anthropic_thinking_block_started, anthropic_text_block_started)

    if (first) {
        events.push_back({
            {"event", "message_start"},
            {"data", {
                {"type", "message_start"},
                {"message", {
                    {"id", oaicompat_cmpl_id},
                    {"type", "message"},
                    {"role", "assistant"},
                    {"content", json::array()},
                    {"model", oaicompat_model},
                    {"stop_reason", nullptr},
                    {"stop_sequence", nullptr},
                    {"usage", {
                        {"cache_read_input_tokens", n_prompt_tokens_cache},
                        {"input_tokens", n_prompt_tokens - n_prompt_tokens_cache},
                        {"output_tokens", 0}
                    }}
                }}
            }}
        });
    }

    // content block indices: thinking (0) -> text (0 or 1) -> tool_use (n+)
    size_t thinking_block_index = 0;
    // use anthropic_has_reasoning (set in update()) to know if ANY reasoning was generated
    size_t text_block_index     = anthropic_has_reasoning ? 1 : 0;

    // use local copies of streaming state (copied from task_result_state in update())
    // these reflect the state BEFORE this chunk was processed
    bool thinking_started = thinking_block_started;
    bool text_started     = text_block_started;

    for (const auto & diff : oaicompat_msg_diffs) {
        // handle thinking/reasoning content
        if (!diff.reasoning_content_delta.empty()) {
            if (!thinking_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", thinking_block_index},
                        {"content_block", {
                            {"type", "thinking"},
                            {"thinking", ""}
                        }}
                    }}
                });
                thinking_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", thinking_block_index},
                    {"delta", {
                        {"type", "thinking_delta"},
                        {"thinking", diff.reasoning_content_delta}
                    }}
                }}
            });
        }

        // handle regular text content
        if (!diff.content_delta.empty()) {
            if (!text_started) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", text_block_index},
                        {"content_block", {
                            {"type", "text"},
                            {"text", ""}
                        }}
                    }}
                });
                text_started = true;
            }

            events.push_back({
                {"event", "content_block_delta"},
                {"data", {
                    {"type", "content_block_delta"},
                    {"index", text_block_index},
                    {"delta", {
                        {"type", "text_delta"},
                        {"text", diff.content_delta}
                    }}
                }}
            });
        }

        // handle tool calls
        if (diff.tool_call_index != std::string::npos) {
            // use anthropic_has_reasoning for thinking block count (persists across calls)
            size_t content_block_index = (anthropic_has_reasoning ? 1 : 0) + (text_started ? 1 : 0) + diff.tool_call_index;

            if (!diff.tool_call_delta.name.empty()) {
                events.push_back({
                    {"event", "content_block_start"},
                    {"data", {
                        {"type", "content_block_start"},
                        {"index", content_block_index},
                        {"content_block", {
                            {"type", "tool_use"},
                            {"id", diff.tool_call_delta.id},
                            {"name", diff.tool_call_delta.name}
                        }}
                    }}
                });
            }

            if (!diff.tool_call_delta.arguments.empty()) {
                events.push_back({
                    {"event", "content_block_delta"},
                    {"data", {
                        {"type", "content_block_delta"},
                        {"index", content_block_index},
                        {"delta", {
                            {"type", "input_json_delta"},
                            {"partial_json", diff.tool_call_delta.arguments}
                        }}
                    }}
                });
            }
        }
    }

    return events;
}

//
// server_task_result_embd
//
json server_task_result_embd::to_json() {
    return res_type == TASK_RESPONSE_TYPE_OAI_EMBD
        ? to_json_oaicompat()
        : to_json_non_oaicompat();
}

json server_task_result_embd::to_json_non_oaicompat() {
    return json {
        {"index",     index},
        {"embedding", embedding},
    };
}

json server_task_result_embd::to_json_oaicompat() {
    return json {
        {"index",            index},
        {"embedding",        embedding[0]},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_tts
//
json server_task_result_tts::to_json() {
    return json {
        {"frames",     n_frames},
        {"sample_rate", sample_rate},
        {"samples",    n_samples},
        {"timings", {
            {"prompt_s",  t_prompt_s},
            {"gen_s",     t_gen_s},
            {"vocoder_s", t_vocoder_s},
            {"total_s",   t_total_s},
        }},
    };
}

//
// server_task_result_rerank
//
json server_task_result_rerank::to_json() {
    return json {
        {"index",            index},
        {"score",            score},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_decision
//
json server_task_result_decision::to_json() {
    return json {
        {"index",            index},
        {"scores",           scores},
        {"tokens_evaluated", n_tokens},
    };
}

//
// server_task_result_error
//
json server_task_result_error::to_json() {
    json res = format_error_response(err_msg, err_type);
    if (err_type == ERROR_TYPE_EXCEED_CONTEXT_SIZE) {
        res["n_prompt_tokens"] = n_prompt_tokens;
        res["n_ctx"]           = n_ctx;
    }
    return res;
}

//
// server_task_result_metrics
//
json server_task_result_slots::to_json() {
    return slots_data;
}

json server_task_result_metrics::to_json() {
    // not used, /metrics renders prometheus text via to_metrics()
    return json{};
}

// metrics definition: https://prometheus.io/docs/practices/naming/#metric-names
std::string server_task_result_metrics::to_metrics() {
    const std::vector<metric_item> counters = {
        {
            "prompt_tokens_total",
            "Number of prompt tokens processed, excluding cached tokens",
            (double) metrics.prompt.count
        }, {
            "prompt_tokens_cached_total",
            "Number of prompt tokens reused from the cache",
            (double) metrics.n_prompt_cached
        }, {
            "prompt_seconds_total",
            "Total time spent processing prompts",
            metrics.prompt.time / 1.e6
        }, {
            "tokens_predicted_total",
            "Number of generation tokens processed",
            (double) metrics.predict.count
        }, {
            "tokens_predicted_seconds_total",
            "Total time spent generating tokens",
            metrics.predict.time / 1.e6
        }, {
            "n_decode_total",
            "Total number of llama_decode() calls, excluding speculative decoding and multimodal decoding",
            (double) metrics.n_decode
        }, {
            "n_tokens_max",
            "Largest observed sequence length (prompt + generation)",
            (double) metrics.n_tokens_max
        }, {
            "spec_decode_num_draft_tokens_total",
            "Speculative: Total draft tokens generated",
            (double) metrics.n_draft_tokens
        }, {
            "spec_decode_num_accepted_tokens_total",
            "Speculative: Total draft tokens accepted by the target model",
            (double) metrics.n_draft_accepted
        }, {
            "spec_decode_num_drafts_total",
            "Speculative: Total speculative decoding verification steps",
            (double) metrics.n_draft_verif_steps
        },
    };

    const std::vector<metric_item> gauges = {
        {
            "prompt_tokens_seconds",
            "Average prompt throughput in tokens/s",
            metrics.prompt_bucket.n_per_second()
        }, {
            "predicted_tokens_seconds",
            "Average generation throughput in tokens/s",
            metrics.predict_bucket.n_per_second()
        }, {
            "requests_processing",
            "Number of requests processing",
            (double) n_processing_slots
        }, {
            "requests_deferred",
            "Number of requests deferred",
            (double) n_tasks_deferred
        }, {
            "n_busy_slots_per_decode",
            "Average number of busy slots per llama_decode() call",
            (double) metrics.n_busy_slots / std::max((double) metrics.n_decode, 1.0)
        },
    };

    std::stringstream prometheus;

    auto add_items = [&prometheus](const char * type, const std::vector<metric_item> & items) {
        for (const auto & item : items) {
            prometheus << "# HELP llamacpp:" << item.name << " " << item.description << "\n"
                       << "# TYPE llamacpp:" << item.name << " " << type             << "\n"
                       << "llamacpp:"        << item.name << " " << item.value       << "\n";
        }
    };

    add_items("counter", counters);
    add_items("gauge",   gauges);

    // labeled counter: one time series per draft position
    if (!metrics.n_accepted_per_pos.empty()) {
        prometheus << "# HELP llamacpp:spec_decode_num_accepted_tokens_per_pos_total"
                      " Accepted tokens per draft position\n"
                   << "# TYPE llamacpp:spec_decode_num_accepted_tokens_per_pos_total counter\n";
        for (size_t i = 0; i < metrics.n_accepted_per_pos.size(); i++) {
            prometheus << "llamacpp:spec_decode_num_accepted_tokens_per_pos_total{position=\""
                       << i << "\"} " << metrics.n_accepted_per_pos[i] << "\n";
        }
    }

    return prometheus.str();
}

//
// server_task_result_slot_save_load
//
json server_task_result_slot_save_load::to_json() {
    if (is_save) {
        return json {
            { "id_slot",   id_slot },
            { "filename",  filename },
            { "n_saved",   n_tokens },
            { "n_written", n_bytes },
            { "timings", {
                { "save_ms", t_ms }
            }},
        };
    }

    return json {
        { "id_slot",    id_slot },
        { "filename",   filename },
        { "n_restored", n_tokens },
        { "n_read",     n_bytes },
        { "timings", {
            { "restore_ms", t_ms }
        }},
    };
}

//
// server_task_result_slot_erase
//
json server_task_result_slot_erase::to_json() {
    return json {
        { "id_slot",  id_slot },
        { "n_erased", n_erased },
    };
}

//
// server_task_result_get_lora
//

json server_task_result_get_lora::to_json() {
    json result = json::array();
    for (size_t i = 0; i < loras.size(); ++i) {
        auto & lora = loras[i];
        json entry = {
            {"id",            i},
            {"path",          lora.info.path},
            {"scale",         lora.info.scale},
            {"task_name",     lora.info.task_name},
            {"prompt_prefix", lora.info.prompt_prefix},
        };
        if (!lora.alora_invocation_tokens.empty()) {
            entry["alora_invocation_string"] = lora.alora_invocation_string;
            entry["alora_invocation_tokens"] = lora.alora_invocation_tokens;
        }
        result.push_back(std::move(entry));
    }
    return result;
}

//
// server_task_result_apply_lora
//

json server_task_result_apply_lora::to_json() {
    return json {{ "success", true }};
}

namespace {

// small self-contained MD5, used only to fingerprint the kernel manifest in the disk-cache index
struct md5_ctx {
    uint32_t state[4] = { 0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476 };
    uint64_t count = 0;
    uint8_t  buffer[64] = {};
};

uint32_t md5_rol(uint32_t v, uint32_t n) {
    return (v << n) | (v >> (32 - n));
}

void md5_transform(md5_ctx & ctx, const uint8_t block[64]) {
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613, 0xfd469501,
        0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821,
        0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a,
        0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70,
        0x289b7ec6, 0xeaa127fa, 0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391,
    };
    static const uint32_t S[64] = {
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
        5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20, 5,  9, 14, 20,
        4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
        6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    };

    uint32_t M[16];
    for (uint32_t i = 0; i < 16; ++i) {
        M[i] = (uint32_t) block[i*4 + 0]
             | (uint32_t) block[i*4 + 1] << 8
             | (uint32_t) block[i*4 + 2] << 16
             | (uint32_t) block[i*4 + 3] << 24;
    }

    uint32_t A = ctx.state[0];
    uint32_t B = ctx.state[1];
    uint32_t C = ctx.state[2];
    uint32_t D = ctx.state[3];

    for (uint32_t i = 0; i < 64; ++i) {
        uint32_t F = 0;
        uint32_t g = 0;
        if (i < 16) {
            F = (B & C) | (~B & D);
            g = i;
        } else if (i < 32) {
            F = (D & B) | (~D & C);
            g = (5*i + 1) % 16;
        } else if (i < 48) {
            F = B ^ C ^ D;
            g = (3*i + 5) % 16;
        } else {
            F = C ^ (B | ~D);
            g = (7*i) % 16;
        }
        const uint32_t tmp = D;
        D = C;
        C = B;
        B = B + md5_rol(A + F + K[i] + M[g], S[i]);
        A = tmp;
    }

    ctx.state[0] += A;
    ctx.state[1] += B;
    ctx.state[2] += C;
    ctx.state[3] += D;
}

void md5_update(md5_ctx & ctx, const uint8_t * data, size_t size) {
    while (size > 0) {
        const size_t n = std::min<size_t>(size, sizeof(ctx.buffer) - (ctx.count % sizeof(ctx.buffer)));
        std::memcpy(ctx.buffer + ctx.count % sizeof(ctx.buffer), data, n);
        ctx.count += n;
        data += n;
        size -= n;
        if (ctx.count % sizeof(ctx.buffer) == 0) {
            md5_transform(ctx, ctx.buffer);
        }
    }
}

std::string md5_final(md5_ctx & ctx) {
    const uint64_t bits = ctx.count * 8;
    const uint8_t one = 0x80;
    md5_update(ctx, &one, 1);
    const uint8_t zero = 0;
    while (ctx.count % sizeof(ctx.buffer) != 56) {
        md5_update(ctx, &zero, 1);
    }
    uint8_t length[8];
    for (int i = 0; i < 8; ++i) {
        length[i] = (uint8_t) (bits >> (8*i));
    }
    md5_update(ctx, length, sizeof(length));

    static const char hex[] = "0123456789abcdef";
    std::string res(32, '0');
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            const uint8_t b = (uint8_t) (ctx.state[i] >> (8*j));
            res[(i*4 + j)*2 + 0] = hex[b >> 4];
            res[(i*4 + j)*2 + 1] = hex[b & 0xf];
        }
    }
    return res;
}

static constexpr uint32_t PROMPT_CACHE_FILE_MAGIC   = 0x53445053; // "SPDS"
static constexpr uint32_t PROMPT_CACHE_FILE_VERSION = 1;

template <typename T>
void pc_write(std::ostream & os, const T & v) {
    os.write(reinterpret_cast<const char *>(&v), sizeof(v));
}

size_t pc_file_size(size_t size_main, size_t size_drft, const std::list<common_prompt_checkpoint> & checkpoints) {
    size_t res = sizeof(uint32_t) * 2 + sizeof(uint64_t) * 2 + sizeof(uint32_t);
    for (const auto & ckpt : checkpoints) {
        res += sizeof(int64_t) + sizeof(int32_t) * 2;
        res += sizeof(uint64_t) * 3 + ckpt.data_tgt.size() + ckpt.data_dft.size() + ckpt.data_spec.size();
    }
    return res + size_main + size_drft;
}

bool pc_write_file(const std::string & path, const server_prompt_data & data) {
    const std::string tmp = path + ".tmp";
    std::ofstream ofs(std::filesystem::u8path(tmp), std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return false;
    }

    const uint32_t magic   = PROMPT_CACHE_FILE_MAGIC;
    const uint32_t version = PROMPT_CACHE_FILE_VERSION;
    const uint64_t size_main = data.main.size();
    const uint64_t size_drft = data.drft.size();
    const uint32_t n_ckpt    = (uint32_t) data.checkpoints.size();
    pc_write(ofs, magic);
    pc_write(ofs, version);
    pc_write(ofs, size_main);
    pc_write(ofs, size_drft);
    pc_write(ofs, n_ckpt);
    for (const auto & ckpt : data.checkpoints) {
        const uint64_t n_tgt  = ckpt.data_tgt.size();
        const uint64_t n_dft  = ckpt.data_dft.size();
        const uint64_t n_spec = ckpt.data_spec.size();
        pc_write(ofs, ckpt.n_tokens);
        pc_write(ofs, ckpt.pos_min);
        pc_write(ofs, ckpt.pos_max);
        pc_write(ofs, n_tgt);
        pc_write(ofs, n_dft);
        pc_write(ofs, n_spec);
        if (n_tgt)  { ofs.write((const char *) ckpt.data_tgt.data(),  n_tgt); }
        if (n_dft)  { ofs.write((const char *) ckpt.data_dft.data(),  n_dft); }
        if (n_spec) { ofs.write((const char *) ckpt.data_spec.data(), n_spec); }
    }
    if (size_main) { ofs.write((const char *) data.main.data(), size_main); }
    if (size_drft) { ofs.write((const char *) data.drft.data(), size_drft); }
    ofs.flush();
    if (!ofs) {
        ofs.close();
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(tmp), ec);
        return false;
    }
    ofs.close();

    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(path), ec);
    if (ec) {
        std::filesystem::remove(std::filesystem::u8path(tmp), ec);
        return false;
    }
    return true;
}

bool pc_read_file(const std::string & path, server_prompt_data & data, size_t expected_size) {
    std::ifstream ifs(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!ifs) {
        return false;
    }
    const std::streamoff file_size = ifs.tellg();
    if (file_size < 0 || (size_t) file_size != expected_size) {
        return false;
    }
    ifs.seekg(0);

    uint32_t magic = 0;
    uint32_t version = 0;
    uint64_t size_main = 0;
    uint64_t size_drft = 0;
    uint32_t n_ckpt = 0;
    ifs.read((char *) &magic, sizeof(magic));
    ifs.read((char *) &version, sizeof(version));
    ifs.read((char *) &size_main, sizeof(size_main));
    ifs.read((char *) &size_drft, sizeof(size_drft));
    ifs.read((char *) &n_ckpt, sizeof(n_ckpt));
    if (!ifs || magic != PROMPT_CACHE_FILE_MAGIC || version != PROMPT_CACHE_FILE_VERSION) {
        return false;
    }

    auto read_buf = [&](std::vector<uint8_t> & buf, uint64_t n) {
        const std::streamoff pos = ifs.tellg();
        if (pos < 0 || n > (uint64_t) (file_size - pos)) {
            return false;
        }
        buf.resize((size_t) n);
        if (n) {
            ifs.read((char *) buf.data(), n);
        }
        return (bool) ifs;
    };

    data.checkpoints.clear();
    for (uint32_t i = 0; i < n_ckpt; ++i) {
        int64_t n_tokens = 0;
        int32_t pos_min = 0;
        int32_t pos_max = 0;
        uint64_t n_tgt = 0;
        uint64_t n_dft = 0;
        uint64_t n_spec = 0;
        ifs.read((char *) &n_tokens, sizeof(n_tokens));
        ifs.read((char *) &pos_min, sizeof(pos_min));
        ifs.read((char *) &pos_max, sizeof(pos_max));
        ifs.read((char *) &n_tgt, sizeof(n_tgt));
        ifs.read((char *) &n_dft, sizeof(n_dft));
        ifs.read((char *) &n_spec, sizeof(n_spec));
        if (!ifs) {
            return false;
        }
        common_prompt_checkpoint ckpt;
        ckpt.id_task  = -1;
        ckpt.n_tokens = n_tokens;
        ckpt.pos_min  = pos_min;
        ckpt.pos_max  = pos_max;
        if (!read_buf(ckpt.data_tgt, n_tgt) || !read_buf(ckpt.data_dft, n_dft) || !read_buf(ckpt.data_spec, n_spec)) {
            return false;
        }
        if (ckpt.data_tgt.empty()) {
            return false;
        }
        data.checkpoints.push_back(std::move(ckpt));
    }
    if (!read_buf(data.main, size_main) || !read_buf(data.drft, size_drft)) {
        return false;
    }
    return ifs.tellg() == file_size;
}

} // namespace

std::string server_file_md5(const std::string & path) {
    std::ifstream ifs(std::filesystem::u8path(path), std::ios::binary);
    if (!ifs) {
        return std::string();
    }
    md5_ctx ctx;
    char buf[64*1024];
    while (ifs) {
        ifs.read(buf, sizeof(buf));
        const std::streamsize n = ifs.gcount();
        if (n > 0) {
            md5_update(ctx, (const uint8_t *) buf, (size_t) n);
        }
    }
    return md5_final(ctx);
}

//
// server_prompt_cache
//
size_t server_prompt_cache::size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return size_locked();
}

size_t server_prompt_cache::size_locked() const {
    size_t res = 0;

    for (const auto & state : states) {
        if (state->keep_ram) {
            res += state->size();
        }
    }

    return res;
}

size_t server_prompt_cache::disk_size() const {
    std::lock_guard<std::mutex> lock(mutex);
    return disk_size_locked();
}

size_t server_prompt_cache::disk_size_locked() const {
    size_t res = 0;

    for (const auto & state : states) {
        if (!state->file.empty()) {
            res += state->file_size;
        }
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    std::lock_guard<std::mutex> lock(mutex);
    return n_tokens_locked();
}

size_t server_prompt_cache::n_tokens_locked() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state->prompt.n_tokens();
    }

    return res;
}

void server_prompt_cache::init_disk(const std::string & disk_path) {
    disk_dir = disk_path;
    if (disk_dir.empty() || disk_dir.back() != DIRECTORY_SEPARATOR) {
        disk_dir += DIRECTORY_SEPARATOR;
    }

    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::u8path(disk_dir), ec);
    if (ec) {
        SRV_WRN("failed to create prompt cache directory '%s': %s\n", disk_dir.c_str(), ec.message().c_str());
        disk_dir.clear();
        return;
    }

    disk_index = disk_dir + "index.json";
    worker = std::thread([this]() { worker_loop(); });

    load_index();
    update();
}

void server_prompt_cache::load_index() {
    auto remove_all = [this]() {
        std::error_code ec;
        for (const auto & entry : std::filesystem::directory_iterator(std::filesystem::u8path(disk_dir), ec)) {
            if (entry.is_regular_file(ec)) {
                std::filesystem::remove(entry.path(), ec);
            }
        }
    };

    std::ifstream ifs(std::filesystem::u8path(disk_index), std::ios::binary);
    if (!ifs) {
        remove_all();
        return;
    }
    std::stringstream ss;
    ss << ifs.rdbuf();

    json index;
    try {
        index = json::parse(ss.str());
    } catch (const std::exception & err) {
        SRV_WRN("ignoring invalid prompt cache index '%s': %s\n", disk_index.c_str(), err.what());
        remove_all();
        return;
    }
    if (!index.is_object() || !index.contains("identity") || !index.at("identity").is_object() ||
        !index.contains("states") || !index.at("states").is_array()) {
        SRV_WRN("ignoring invalid prompt cache index '%s'\n", disk_index.c_str());
        remove_all();
        return;
    }

    bool identity_ok = false;
    try {
        identity_ok =
            index.value("version", 0) == 1 &&
            index.at("identity").value("model_path", std::string()) == identity.model_path &&
            index.at("identity").value("model_size", uint64_t(0)) == identity.model_size &&
            index.at("identity").value("mmproj_path", std::string()) == identity.mmproj_path &&
            index.at("identity").value("mmproj_size", uint64_t(0)) == identity.mmproj_size &&
            index.at("identity").value("n_ctx", int32_t(0)) == identity.n_ctx &&
            index.at("identity").value("kernel_manifest_md5", std::string()) == identity.kernel_manifest_md5 &&
            index.at("identity").value("has_draft", false) == identity.has_draft;
    } catch (const std::exception & err) {
        SRV_WRN("ignoring invalid prompt cache identity in '%s': %s\n", disk_index.c_str(), err.what());
        remove_all();
        return;
    }

    const json & entries = index.at("states");
    if (!identity_ok) {
        SRV_WRN("prompt cache identity changed, deleting old disk states in '%s'\n", disk_dir.c_str());
        remove_all();
        return;
    }

    std::set<std::string> keep_files;
    for (const auto & entry : entries) {
        if (!entry.is_object()) {
            continue;
        }
        std::string file;
        uint64_t file_size = 0;
        try {
            file      = entry.value("file", std::string());
            file_size = entry.value("size", uint64_t(0));
        } catch (const std::exception & err) {
            SRV_WRN("ignoring invalid prompt cache index entry: %s\n", err.what());
            continue;
        }
        if (file.empty() || file.find('/') != std::string::npos || file.find('\\') != std::string::npos) {
            continue;
        }

        std::error_code ec;
        const std::string filepath = disk_dir + file;
        if (!std::filesystem::exists(std::filesystem::u8path(filepath), ec) ||
            std::filesystem::file_size(std::filesystem::u8path(filepath), ec) != file_size) {
            std::filesystem::remove(std::filesystem::u8path(filepath), ec);
            continue;
        }

        try {
            const std::vector<int> packed_int = entry.at("tokens").get<std::vector<int>>();
            llama_tokens packed;
            packed.reserve(packed_int.size());
            for (const int token : packed_int) {
                packed.push_back((llama_token) token);
            }
            server_tokens tokens = server_tokens::deserialize(packed, has_mtmd);
            if (tokens.empty()) {
                continue;
            }

            keep_files.insert(file);

            auto state = std::make_shared<server_prompt_cache_state>();
            state->id         = next_id++;
            state->prompt.tokens = std::move(tokens);
            state->file       = file;
            state->file_size  = (size_t) file_size;
            state->disk_ready = true;
            state->keep_ram   = false;

            if (entry.contains("checkpoints") && entry.at("checkpoints").is_array()) {
                for (const auto & ckpt : entry.at("checkpoints")) {
                    common_prompt_checkpoint c;
                    c.id_task  = -1;
                    c.n_tokens = ckpt.value("n_tokens", int64_t(0));
                    c.pos_min  = ckpt.value("pos_min", int32_t(0));
                    c.pos_max  = ckpt.value("pos_max", int32_t(0));
                    state->prompt.checkpoints.push_back(std::move(c));
                }
            }
            states.push_back(std::move(state));
        } catch (const std::exception & err) {
            SRV_WRN("ignoring prompt cache index entry '%s': %s\n", file.c_str(), err.what());
            std::error_code ec;
            std::filesystem::remove(std::filesystem::u8path(disk_dir + file), ec);
        }
    }

    std::error_code ec;
    for (const auto & entry : std::filesystem::directory_iterator(std::filesystem::u8path(disk_dir), ec)) {
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.compare(0, 6, "state-") == 0 && keep_files.count(name) == 0) {
            std::filesystem::remove(entry.path(), ec);
        }
    }

    SRV_TRC("prompt cache: loaded %zu disk state(s), %.3f MiB from '%s'\n",
            states.size(), disk_size_locked() / (1024.0 * 1024.0), disk_dir.c_str());
}

void server_prompt_cache::write_index() const {
    if (disk_dir.empty()) {
        return;
    }

    json index = json::object();
    index["version"] = 1;
    index["identity"] = json::object();
    index["identity"]["model_path"] = identity.model_path;
    index["identity"]["model_size"] = identity.model_size;
    index["identity"]["mmproj_path"] = identity.mmproj_path;
    index["identity"]["mmproj_size"] = identity.mmproj_size;
    index["identity"]["n_ctx"] = identity.n_ctx;
    index["identity"]["kernel_manifest_md5"] = identity.kernel_manifest_md5;
    index["identity"]["has_draft"] = identity.has_draft;

    json entries = json::array();
    for (const auto & state : states) {
        if (state->file.empty()) {
            continue;
        }

        json entry = json::object();
        entry["file"] = state->file;
        entry["size"] = (uint64_t) state->file_size;

        json tokens = json::array();
        try {
            const std::vector<char> packed = state->prompt.tokens.serialize();
            for (size_t i = 0; i + sizeof(uint32_t) <= packed.size(); i += sizeof(uint32_t)) {
                uint32_t token = 0;
                std::memcpy(&token, packed.data() + i, sizeof(token));
                tokens.push_back((int) token);
            }
        } catch (const std::exception & err) {
            SRV_WRN("failed to serialize prompt tokens for disk cache '%s': %s\n", state->file.c_str(), err.what());
            continue;
        }
        entry["tokens"] = tokens;

        json checkpoints = json::array();
        for (const auto & ckpt : state->prompt.checkpoints) {
            json c = json::object();
            c["n_tokens"] = (int64_t) ckpt.n_tokens;
            c["pos_min"]  = (int32_t) ckpt.pos_min;
            c["pos_max"]  = (int32_t) ckpt.pos_max;
            checkpoints.push_back(c);
        }
        entry["checkpoints"] = checkpoints;
        entries.push_back(entry);
    }
    index["states"] = entries;

    const std::string tmp = disk_index + ".tmp";
    std::ofstream ofs(std::filesystem::u8path(tmp), std::ios::binary | std::ios::trunc);
    if (!ofs) {
        return;
    }
    const std::string body = index.dump();
    ofs.write(body.data(), body.size());
    ofs.flush();
    if (!ofs) {
        ofs.close();
        std::error_code ec;
        std::filesystem::remove(std::filesystem::u8path(tmp), ec);
        return;
    }
    ofs.close();
    std::error_code ec;
    std::filesystem::rename(std::filesystem::u8path(tmp), std::filesystem::u8path(disk_index), ec);
}

void server_prompt_cache::remove_state(std::list<std::shared_ptr<server_prompt_cache_state>>::iterator it) {
    const auto & state = *it;
    if (disk_enabled() && !state->file.empty()) {
        job j;
        j.type     = job::REMOVE;
        j.id       = state->id;
        j.filepath = disk_dir + state->file;
        enqueue(std::move(j));
    }
    states.erase(it);
}

void server_prompt_cache::enqueue(job && j) {
    jobs.push_back(std::move(j));
    cv.notify_one();
}

bool server_prompt_cache::save_file(const job & j) const {
    return j.data && pc_write_file(j.filepath, *j.data);
}

void server_prompt_cache::worker_loop() {
    while (true) {
        job j;
        {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock, [this]() { return stop || !jobs.empty(); });
            if (stop && jobs.empty()) {
                break;
            }
            j = std::move(jobs.front());
            jobs.pop_front();
        }

        if (j.type == job::SAVE) {
            const bool ok = save_file(j);
            std::lock_guard<std::mutex> lock(mutex);
            auto it = std::find_if(states.begin(), states.end(), [&](const auto & state) { return state->id == j.id; });
            if (ok) {
                std::error_code ec;
                const uint64_t file_size = std::filesystem::file_size(std::filesystem::u8path(j.filepath), ec);
                if (it != states.end()) {
                    (*it)->disk_ready = true;
                    (*it)->file_size  = ec ? (*it)->file_size : (size_t) file_size;
                    if (!(*it)->keep_ram) {
                        (*it)->data.reset();
                    }
                } else {
                    std::filesystem::remove(std::filesystem::u8path(j.filepath), ec);
                }
            } else {
                SRV_WRN("failed to write prompt cache state '%s'\n", j.filepath.c_str());
                if (it != states.end()) {
                    if ((*it)->keep_ram) {
                        (*it)->file.clear();
                        (*it)->file_size = 0;
                    } else {
                        states.erase(it);
                    }
                }
            }
            write_index();
        } else {
            std::error_code ec;
            std::filesystem::remove(std::filesystem::u8path(j.filepath), ec);
            std::lock_guard<std::mutex> lock(mutex);
            write_index();
        }
    }
}

server_prompt_cache::~server_prompt_cache() {
    {
        std::lock_guard<std::mutex> lock(mutex);
        stop = true;
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

server_prompt_cache_state * server_prompt_cache::alloc(server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    std::lock_guard<std::mutex> lock(mutex);

    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = (*it)->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size()) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt.size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;
    const size_t file_size_new  = disk_enabled() ? pc_file_size(state_size_tgt, state_size_dft, prompt.checkpoints) : 0;

    // skip over-limit entries to avoid disturbing the cache
    if (ram_enabled && limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }
    if (disk_enabled() && limit_disk > 0 && file_size_new > limit_disk) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds disk cache size limit %.3f MiB, skipping\n",
                file_size_new / (1024.0 * 1024.0), limit_disk / (1024.0 * 1024.0));
        return nullptr;
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = (*it)->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) (*it)->prompt.tokens.size()) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            remove_state(it++);
        } else {
            ++it;
        }
    }

    if (ram_enabled && limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit
        while (size_locked() + state_size_new > limit_size) {
            auto it = std::find_if(states.begin(), states.end(), [](const auto & state) {
                return state->keep_ram && state->data;
            });
            if (it == states.end()) {
                break;
            }

            if (disk_enabled() && !(*it)->file.empty() && (*it)->disk_ready) {
                SRV_WRN(" - making room for prompt cache entry, spilling oldest entry to disk (size = %.3f MiB)\n",
                        (*it)->size() / (1024.0 * 1024.0));
                (*it)->data.reset();
            } else {
                SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                        (*it)->size() / (1024.0 * 1024.0));
                remove_state(it);
            }
        }
    }

    if (disk_enabled() && limit_disk > 0) {
        while (disk_size_locked() + file_size_new > limit_disk) {
            auto it = std::find_if(states.begin(), states.end(), [](const auto & state) {
                return !state->file.empty();
            });
            if (it == states.end()) {
                break;
            }

            if ((*it)->keep_ram) {
                SRV_WRN(" - disk cache limit reached, deleting oldest entry '%s'\n", (*it)->file.c_str());
                job j;
                j.type     = job::REMOVE;
                j.id       = (*it)->id;
                j.filepath = disk_dir + (*it)->file;
                enqueue(std::move(j));
                (*it)->file.clear();
                (*it)->file_size  = 0;
                (*it)->disk_ready = false;
            } else {
                SRV_WRN(" - disk cache limit reached, removing oldest entry '%s'\n", (*it)->file.c_str());
                remove_state(it);
            }
        }
    }

    auto state = std::make_shared<server_prompt_cache_state>();
    state->id         = next_id++;
    state->keep_ram   = ram_enabled;
    state->disk_ready = false;

    try {
        state->data = std::make_shared<server_prompt_data>();
        state->data->main.resize(state_size_tgt);
        state->data->drft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        if (!ram_enabled) {
            return nullptr;
        }

        limit_size = std::max<size_t>(1, 0.4*size_locked());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        while (!states.empty() && size_locked() + state_size_new > limit_size) {
            remove_state(states.begin());
        }

        return nullptr;
    }

    state->prompt.tokens = prompt.tokens.clone();
    // Keep only checkpoint metadata in the prompt entry.  The payload is
    // already stored in data->checkpoints and is moved back on load.
    state->prompt.checkpoints.clear();
    for (const auto & ckpt : prompt.checkpoints) {
        common_prompt_checkpoint info;
        info.id_task  = ckpt.id_task;
        info.n_tokens = ckpt.n_tokens;
        info.pos_min  = ckpt.pos_min;
        info.pos_max  = ckpt.pos_max;
        state->prompt.checkpoints.push_back(std::move(info));
    }

    // take over the payloads instead of copying them: the slot still owns its
    // copy until prompt_load() replaces its prompt, so a copy would briefly
    // double the checkpoint memory outside the cache budget
    state->data->checkpoints = std::move(prompt.checkpoints);

    if (disk_enabled()) {
        std::ostringstream ss;
        ss << "state-" << std::chrono::steady_clock::now().time_since_epoch().count() << '-' << state->id << ".bin";
        state->file      = ss.str();
        state->file_size = file_size_new;
    }

    states.push_back(std::move(state));

    return states.back().get();
}

bool server_prompt_cache::store_disk(const server_prompt_cache_state & state) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!disk_enabled() || state.file.empty() || !state.data) {
        return false;
    }

    job j;
    j.type     = job::SAVE;
    j.id       = state.id;
    j.filepath = disk_dir + state.file;
    j.data     = state.data;
    enqueue(std::move(j));
    return true;
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    std::lock_guard<std::mutex> lock(mutex);

    const int lcp_best = prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = prompt.tokens.size() > 0 ? float(lcp_best) / prompt.tokens.size() : -1.0f; // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int lcp_cur = (*it)->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / (*it)->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", (*it)->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;

            it_best = it;
        }
    }

    if (it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        std::shared_ptr<server_prompt_data> data = (*it_best)->data;
        if (!data && (*it_best)->disk_ready) {
            data = std::make_shared<server_prompt_data>();
            if (!pc_read_file(disk_dir + (*it_best)->file, *data, (*it_best)->file_size)) {
                SRV_WRN("failed to read prompt cache state '%s'\n", (*it_best)->file.c_str());
                remove_state(it_best);
                return false;
            }
        }

        if (!data) {
            SRV_WRN("prompt cache state '%s' has no data\n", (*it_best)->file.c_str());
            remove_state(it_best);
            return false;
        }

        if ((ctx_dft != nullptr) != !data->drft.empty()) {
            SRV_WRN("prompt cache state '%s' does not match the draft context\n", (*it_best)->file.c_str());
            remove_state(it_best);
            return false;
        }

        {
            const size_t size = data->main.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data->main.data(), size, id_slot, 0);
            if (n != size) {
                SRV_ERR("failed to restore state with size %zu\n", size);
                remove_state(it_best);
                return false;
            }
        }

        if (!data->drft.empty()) {
            const size_t size = data->drft.size();
            const size_t n = llama_state_seq_set_data_ext(ctx_dft, data->drft.data(), size, id_slot, 0);
            if (n != size) {
                SRV_WRN("failed to restore state with size %zu\n", size);
                remove_state(it_best);
                return false;
            }
        }

        // a disk-only entry costs nothing in RAM; keep it so that another
        // restart, slot or sequence can reuse the same file
        const bool keep_disk = (*it_best)->disk_ready && !(*it_best)->keep_ram;
        if (keep_disk) {
            prompt.tokens = (*it_best)->prompt.tokens.clone();
        } else {
            prompt = std::move((*it_best)->prompt);
        }
        if (disk_enabled() && !(*it_best)->disk_ready) {
            // the async writer still owns this state; copy the checkpoints
            // instead of moving them out from under the writer
            prompt.checkpoints = data->checkpoints;
        } else {
            prompt.checkpoints = std::move(data->checkpoints);
        }

        if (keep_disk) {
            // most recently used entries go to the back of the LRU list
            states.splice(states.end(), states, it_best);
        } else {
            remove_state(it_best);
        }
    }

    return true;
}

void server_prompt_cache::update() {
    std::lock_guard<std::mutex> lock(mutex);

    if (ram_enabled && limit_size > 0) {
        while (size_locked() > limit_size) {
            auto it = std::find_if(states.begin(), states.end(), [](const auto & state) {
                return state->keep_ram && state->data;
            });
            if (it == states.end()) {
                break;
            }

            if (disk_enabled() && !(*it)->file.empty() && (*it)->disk_ready) {
                SRV_WRN(" - cache size limit reached, spilling oldest entry to disk (size = %.3f MiB)\n",
                        (*it)->size() / (1024.0 * 1024.0));
                (*it)->data.reset();
            } else {
                SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n",
                        (*it)->size() / (1024.0 * 1024.0));
                remove_state(it);
            }
        }
    }

    if (disk_enabled() && limit_disk > 0) {
        while (disk_size_locked() > limit_disk) {
            auto it = std::find_if(states.begin(), states.end(), [](const auto & state) {
                return !state->file.empty();
            });
            if (it == states.end()) {
                break;
            }

            if ((*it)->keep_ram) {
                SRV_WRN(" - disk cache size limit reached, deleting oldest file '%s'\n", (*it)->file.c_str());
                job j;
                j.type     = job::REMOVE;
                j.id       = (*it)->id;
                j.filepath = disk_dir + (*it)->file;
                enqueue(std::move(j));
                (*it)->file.clear();
                (*it)->file_size  = 0;
                (*it)->disk_ready = false;
            } else {
                SRV_WRN(" - disk cache size limit reached, removing oldest entry '%s'\n", (*it)->file.c_str());
                remove_state(it);
            }
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size_locked() + disk_size_locked()) / (std::max<size_t>(1, n_tokens_locked())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = ram_enabled && limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens_locked() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front()->size() / (1024.0 * 1024.0));

            remove_state(states.begin());
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB RAM, %.3f MiB disk (limits: %.3f MiB RAM, %.3f MiB disk, %zu tokens, %zu est)\n",
            states.size(), size_locked() / (1024.0 * 1024.0), disk_size_locked() / (1024.0 * 1024.0),
            limit_size / (1024.0 * 1024.0), limit_disk / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB RAM, %9.3f MiB disk\n",
                (const void *) state.get(), state->prompt.n_tokens(), state->prompt.checkpoints.size(),
                state->size() / (1024.0 * 1024.0), state->file_size / (1024.0 * 1024.0));
    }
}
