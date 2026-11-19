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

// Translate a physical-core selection into the SDK's affinity bitmap for a kernel compiled for `compiled` cores.
// The AX8860 engine runs a 2-core model on core 0 plus the core named by the bitmap, so {0,1} is 0x3 and
// {0,2}/{0,3} are 0x30/0xc0; 0xf (four VNPU bits) is rejected by the SDK. Returns 0 and sets error when the
// selection cannot be expressed.
uint32_t kernel_affinity_mask(const core_selection & sel, int compiled, std::string & error);

}  // namespace axera
