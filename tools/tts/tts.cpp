#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <cstdio>
#include <cstring>
#include <string>

/**
 * Please note that this is NOT a production-ready binary.
 * It is a playground for trying TTS support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand. Do not add unnecessary complexity. The goal is to have a simple CLI for testing TTS support.
 */

struct tts_sampler_ctx {
    common_sampler * smpl;
    llama_context  * lctx;
};

static llama_token tts_sample_semantic_code(void * user_data) {
    auto * ctx = static_cast<tts_sampler_ctx *>(user_data);
    llama_token t = common_sampler_sample(ctx->smpl, ctx->lctx, -1);
    common_sampler_accept(ctx->smpl, t, true);
    return t;
}

static void print_usage(int, char ** argv) {
    LOG("\nexample usage:\n");
    LOG("\n    %s -m backbone.gguf -mm mmproj.gguf -p \"text to speak\" -o output.wav", argv[0]);
    LOG("\n    %s -hf user/model -p \"text to speak\" -o output.wav\n", argv[0]);
    LOG("\nnote: TTS prompt options depend on the model type, see tts/README.md");
    LOG("\n      use -n to limit the output length");
    LOG("\n      see tts/README.md for per-model usage notes");
    LOG("\n\n");
}

int main(int argc, char ** argv) {
    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_TTS, print_usage)) {
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.prompt.empty()) {
        LOG_ERR("no prompt provided, use -p \"text\"\n");
        return 1;
    }
    if (params.mmproj.path.empty()) {
        LOG_ERR("no mmproj provided, use --mmproj\n");
        return 1;
    }

    // important: keep this file as generic as possible
    //            model-specific logic should be in mtmd-helper-gen or mtmd API

    // always enable embd, so that we can pass hidden states to the audio generation helper
    params.embedding = true;

    llama_backend_init();
    llama_numa_init(params.numa);

    //
    // load backbone model and mmproj
    //

    auto llama_init = common_init_from_params(params);
    llama_model    * model = llama_init->model();
    llama_context  * lctx  = llama_init->context();
    common_sampler * smpl  = llama_init->sampler(0);
    if (!model || !lctx) {
        LOG_ERR("failed to init model/context\n");
        return 1;
    }

    char model_type_buf[64] = {};
    if (llama_model_meta_val_str(model, "qwen3tts.tts_model_type", model_type_buf, sizeof(model_type_buf)) < 0) {
        snprintf(model_type_buf, sizeof(model_type_buf), "base");
    }
    const std::string model_type = model_type_buf;
    if (model_type == "base") {
        if (!params.tts_speaker.empty()) {
            LOG_ERR("--tts-speaker is only supported by CustomVoice models\n");
            return 1;
        }
        if (!params.tts_instruct.empty()) {
            LOG_ERR("--tts-instruct is not supported by Base models\n");
            return 1;
        }
    } else if (model_type == "custom_voice" || model_type == "voice_design") {
        if (!params.tts_speaker_file.empty()) {
            LOG_ERR("--tts-speaker-file is only supported by Base models\n");
            return 1;
        }
        if (!params.tts_ref_text.empty()) {
            LOG_ERR("--tts-ref-text is only supported by Base models\n");
            return 1;
        }
        if (model_type == "custom_voice") {
            if (params.tts_speaker.empty()) {
                LOG_ERR("CustomVoice models require --tts-speaker <name>\n");
                return 1;
            }
        } else {
            if (!params.tts_speaker.empty()) {
                LOG_ERR("--tts-speaker is only supported by CustomVoice models\n");
                return 1;
            }
            if (params.tts_instruct.empty()) {
                LOG_ERR("VoiceDesign models require --tts-instruct \"<voice description>\"\n");
                return 1;
            }
        }
    } else {
        LOG_ERR("unsupported Qwen3-TTS model type '%s'\n", model_type.c_str());
        return 1;
    }

    mtmd_context_params mtmd_params = mtmd_context_params_default();
    mtmd_params.use_gpu = params.mmproj_use_gpu;
    mtmd_params.device  = params.mmproj_device;
    mtmd::context_ptr mctx(mtmd_init_from_file(params.mmproj.path.c_str(), model, mtmd_params));
    if (!mctx) {
        LOG_ERR("failed to load mmproj %s\n", params.mmproj.path.c_str());
        return 1;
    }
    if (mtmd_gen_audio_get_info(mctx.get()).type == MTMD_GEN_AUDIO_TYPE_NONE) {
        LOG_ERR("mmproj does not support audio generation\n");
        return 1;
    }

    //
    // stage 0: process speaker reference file, if any
    //

    mtmd::bitmap_ptr speaker_bitmap;
    if (!params.tts_speaker_file.empty()) {
        auto wrapper = mtmd_helper_bitmap_init_from_file(mctx.get(), params.tts_speaker_file.c_str(), false, mtmd_helper_init_opt_default());
        if (!wrapper.bitmap) {
            LOG_ERR("failed to load speaker file %s\n", params.tts_speaker_file.c_str());
            return 1;
        }
        speaker_bitmap.reset(wrapper.bitmap);
    }

    mtmd_helper::gen_audio gen(lctx, mctx.get());
    mtmd_helper_gen_audio_inp inp{};
    inp.seq_id      = 0;
    inp.prompt      = params.prompt.c_str();
    inp.prompt_len  = params.prompt.size();
    inp.speaker_ref = speaker_bitmap.get();
    inp.ref_text     = params.tts_ref_text.c_str();
    inp.ref_text_len = params.tts_ref_text.size();
    inp.speaker      = params.tts_speaker.c_str();
    inp.instruct     = params.tts_instruct.c_str();
    inp.lang         = params.tts_lang.c_str();
    inp.top_k        = params.sampling.top_k;
    inp.top_p        = params.sampling.top_p;
    inp.seed         = params.sampling.seed;
    inp.out_type     = MTMD_HELPER_GEN_AUDIO_OUTTYPE_WAV;

    // stage 1-3: prompt, semantic generation and waveform decoding
    const int max_new = params.n_predict > 0 ? params.n_predict : 512;
    const char * data = nullptr;
    size_t data_len = 0;
    tts_sampler_ctx sample_ctx{smpl, lctx};
    mtmd_helper_gen_audio_run_result run_result{};
    if (gen.run(&inp, params.n_batch, max_new, tts_sample_semantic_code, &sample_ctx,
                &data, &data_len, &run_result) != 0) {
        LOG_ERR("audio generation failed\n");
        return 1;
    }

    FILE * f = fopen(params.out_file.c_str(), "wb");
    if (!f) {
        LOG_ERR("failed to open %s\n", params.out_file.c_str());
        return 1;
    }
    fwrite(data, 1, data_len, f);
    fclose(f);
    LOG_INF("wrote %s\n", params.out_file.c_str());

    llama_backend_free();
    return 0;
}
