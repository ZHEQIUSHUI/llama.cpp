// The host side of the combined final-RMSNorm + lm-head + greedy-argmax kernel.
//
// The adapter publishes one (id, logit) per output row after the combined head has run. common/sampling.cpp enables
// this only for a pure greedy chain and takes the published rows in order; every other chain reads the full logits
// that the same kernel wrote. The state is process-wide, matching the adapter's single sequence.
#pragma once

#include <cstdint>

namespace axera {

// Returns the combined head's row count (0 when disabled or no combined lm_head role is loaded). `ids` are the
// model's suppress tokens (all -INFINITY logit biases); a suppressed winner forces the full-logits fallback.
int  argmax_configure(bool enabled, const int32_t * ids, int32_t n);
void argmax_reset();
bool argmax_wanted();
bool argmax_publish(const int32_t * ids, const float * logits, int32_t rows);
int  argmax_published();
int  argmax_take(int idx, int32_t * id, float * logit);

}  // namespace axera
