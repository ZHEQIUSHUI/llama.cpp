#include "arg.h"
#include "common.h"
#include "debug.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <string>
#include <vector>

static bool run(llama_context * ctx, const common_params & params) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);

    if (tokens.empty()) {
        LOG_ERR("%s : there are not input tokens to process - (try to provide a prompt with '-p')\n", __func__);
        return false;
    }

    LOG_INF("number of input tokens = %zu\n", tokens.size());
    for (size_t i = 0; i < tokens.size(); ++i) {
        LOG_INF("  %d\n", tokens[i]);
    }

    common_batch batch = common_batch_get_one(ctx, tokens);
    // Request activations for every prompt token, not just the last one.
    for (int32_t i = 0; i < (int32_t) tokens.size(); ++i) {
        batch.set_output(i, true);
    }
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) {
        LOG_ERR("%s : failed to eval\n", __func__);
        return false;
    }

    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    // Emit every prompt token's activations (not just the last), so per-layer
    // and final-layer dumps are full-length and comparable to a reference.
    params.n_outputs_max_per_seq = 0;

    // pass the callback to the backend scheduler
    // it will be executed for each node during the graph computation
    common_debug_cb_user_data cb_data(params, params.tensor_filter);
    params.warmup = false;

    // init
    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        LOG_ERR("%s : failed to init\n", __func__);
        return 1;
    }

    // print system information
    {
        LOG_INF("\n");
        LOG_INF("%s\n", common_params_get_system_info(params).c_str());
        LOG_INF("\n");
    }

    bool OK = run(ctx, params);
    if (!OK) {
        return 1;
    }

    LOG("\n");
    llama_perf_context_print(ctx);

    llama_backend_free();

    return 0;
}
