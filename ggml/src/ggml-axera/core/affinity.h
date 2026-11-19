// Axera NPU core selection: GGML_AXERA_NPU_CORES is a comma-separated list of physical core numbers
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace axera {

struct core_selection {
    bool             enabled   = false;
    std::vector<int> cores;
    uint32_t         vnpu_mask = 0;
};

// Parse "2" or "0,1". Invalid input returns an empty cores list and sets error. An unset variable is disabled.
core_selection parse_npu_cores(const char * value, std::string & error);

// The kernel file names end in _npu<count>. -1 means the name carries no count (e.g. an in-memory helper).
int kernel_compiled_cores(const std::string & file);

}  // namespace axera
