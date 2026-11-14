#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "mtmd.h"
#include "forced-aligner.h"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct cli_options {
    std::string transcript;
    std::string language = "English";
    std::string output = "-";
    int segment_ms = 80;
};

std::string take_value(int & i, int argc, char ** argv, const char * name) {
    if (i + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for ") + name);
    }
    return argv[++i];
}

std::vector<std::string> strip_custom_args(int & argc, char ** argv, cli_options & opt) {
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

void print_usage(int argc, char ** argv) {
    GGML_UNUSED(argc);
    fprintf(stderr,
            "usage: %s -m model.gguf --mmproj mmproj.gguf --audio audio.wav --transcript TEXT [options]\n"
            "  --language NAME       language name (default: English)\n"
            "  --output-json PATH    JSON output path, - for stdout (default: -)\n"
            "  --segment-ms N        timestamp step in ms (default: 80)\n"
            "  -c N -ub N -ngl N     standard llama.cpp options are accepted\n",
            argv[0]);
}

std::vector<unsigned char> read_file(const std::string & path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        throw std::runtime_error("cannot open " + path);
    }
    const std::streamsize size = input.tellg();
    if (size <= 0) {
        throw std::runtime_error("empty file " + path);
    }
    input.seekg(0);
    std::vector<unsigned char> data((size_t) size);
    if (!input.read((char *) data.data(), size)) {
        throw std::runtime_error("cannot read " + path);
    }
    return data;
}

} // namespace

int main(int argc, char ** argv) {
    cli_options opt;
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

        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu = params.mmproj_use_gpu;
        mparams.device = params.mmproj_device;
        mparams.print_timings = false;
        mparams.n_threads = params.cpuparams.n_threads;
        mparams.flash_attn_type = params.flash_attn_type;
        mparams.warmup = false;
        mtmd_context * mctx = mtmd_init_from_file(params.mmproj.path.c_str(), model, mparams);
        if (!mctx) {
            fprintf(stderr, "error: failed to load mmproj\n");
            return 1;
        }

        mtmd_forced_aligner::options options = {
            opt.transcript,
            opt.language,
            opt.segment_ms,
        };
        const std::vector<unsigned char> audio = read_file(params.image[0]);
        const mtmd_forced_aligner::result result = mtmd_forced_aligner::run(
                model, ctx, mctx, audio.data(), audio.size(), options);
        mtmd_free(mctx);
        if (!result.ok) {
            fprintf(stderr, "error: %s\n", result.error.c_str());
            return 1;
        }
        if (opt.output == "-") {
            fputs(result.json.c_str(), stdout);
        } else {
            std::ofstream output(opt.output);
            if (!output) {
                fprintf(stderr, "error: cannot write %s\n", opt.output.c_str());
                return 1;
            }
            output << result.json;
        }
        return 0;
    } catch (const std::exception & e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
