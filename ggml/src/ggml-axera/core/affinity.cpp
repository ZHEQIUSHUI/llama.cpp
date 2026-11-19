#include "affinity.h"

#include <cctype>

namespace axera {

core_selection parse_npu_cores(const char * value, std::string & error) {
    core_selection out;
    error.clear();
    if (value == nullptr) {
        return out;
    }
    out.enabled = true;
    if (*value == '\0') {
        error = "GGML_AXERA_NPU_CORES is set but empty; expected a core number or a comma-separated list";
        return out;
    }

    const std::string s(value);
    size_t            pos = 0;
    while (pos <= s.size()) {
        const size_t comma = s.find(',', pos);
        const size_t end   = comma == std::string::npos ? s.size() : comma;
        size_t       a     = pos;
        size_t       b     = end;
        while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
        if (a == b) {
            error = "GGML_AXERA_NPU_CORES contains an empty core number";
            out.cores.clear();
            out.vnpu_mask = 0;
            return out;
        }
        int core = 0;
        for (size_t i = a; i < b; ++i) {
            if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
                error = "GGML_AXERA_NPU_CORES must contain decimal core numbers separated by commas";
                out.cores.clear();
                out.vnpu_mask = 0;
                return out;
            }
            core = core * 10 + (s[i] - '0');
            if (core > 3) {
                error = "GGML_AXERA_NPU_CORES core number is out of range (AX8860 cores are 0-3)";
                out.cores.clear();
                out.vnpu_mask = 0;
                return out;
            }
        }
        for (int c : out.cores) {
            if (c == core) {
                error = "GGML_AXERA_NPU_CORES lists a core more than once";
                out.cores.clear();
                out.vnpu_mask = 0;
                return out;
            }
        }
        out.cores.push_back(core);
        // The engine affinity bitmap has two VNPU bits per physical core. Selecting both reserves the whole core.
        out.vnpu_mask |= 3u << (2 * core);
        if (comma == std::string::npos) {
            break;
        }
        pos = comma + 1;
    }
    return out;
}

int kernel_compiled_cores(const std::string & file) {
    const size_t p = file.rfind("_npu");
    if (p == std::string::npos) {
        return -1;
    }
    size_t i = p + 4;
    if (i == file.size() || !std::isdigit(static_cast<unsigned char>(file[i]))) {
        return -1;
    }
    int n = 0;
    while (i < file.size() && std::isdigit(static_cast<unsigned char>(file[i]))) {
        n = n * 10 + (file[i] - '0');
        ++i;
    }
    return n;
}

}  // namespace axera
