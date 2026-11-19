#include "../core/affinity.h"

#include <cstdio>
#include <string>

static int failures = 0;

static void expect(const char * value, bool enabled, const std::vector<int> & cores, uint32_t mask, bool bad) {
    std::string    error;
    const auto     got = axera::parse_npu_cores(value, error);
    const bool     ok  = got.enabled == enabled && got.cores == cores && got.vnpu_mask == mask && error.empty() != bad;
    if (!ok) {
        std::fprintf(stderr, "parse(%s): enabled=%d cores=%zu mask=0x%x error=%s\n", value ? value : "<null>",
                     got.enabled, got.cores.size(), got.vnpu_mask, error.c_str());
        ++failures;
    }
}

static void expect_mask(const char * value, int compiled, uint32_t mask, bool bad) {
    std::string    error;
    const auto     got = axera::parse_npu_cores(value, error);
    const uint32_t m   = axera::kernel_affinity_mask(got, compiled, error);
    if (error.empty() == bad || (!bad && m != mask)) {
        std::fprintf(stderr, "mask(%s,%d)=0x%x error=%s, expected 0x%x bad=%d\n", value, compiled, m, error.c_str(), mask,
                     bad);
        ++failures;
    }
}

int main() {
    expect(nullptr, false, {}, 0, false);
    expect("2", true, { 2 }, 0x30, false);
    expect("0,1", true, { 0, 1 }, 0xf, false);
    expect(" 3 ", true, { 3 }, 0xc0, false);
    expect("", true, {}, 0, true);
    expect("1,,2", true, {}, 0, true);
    expect("4", true, {}, 0, true);
    expect("1,1", true, {}, 0, true);
    expect("1,x", true, {}, 0, true);

    expect_mask("0,1", 2, 0x3, false);
    expect_mask("0,2", 2, 0x30, false);
    expect_mask("0,3", 2, 0xc0, false);
    expect_mask("1,2", 2, 0, true);
    expect_mask("0,1", 1, 0xf, false);

    const char * names[] = {
        "qwen35_2b_group12_gggagggaggga_mx8_kvc_c262144_npu1.axmodel",
        "qwen35_2b_prefill256_mx8_kvc_c262144_npu1.axmodel",
        "lm_head_combined_n248320_k2048_c4_npu4.axmodel",
        "qwen35_4b_group16_x_npu2.axmodel",
        "<memory>",
    };
    const int counts[] = { 1, 1, 4, 2, -1 };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        const int got = axera::kernel_compiled_cores(names[i]);
        if (got != counts[i]) {
            std::fprintf(stderr, "compiled_cores(%s)=%d, expected %d\n", names[i], got, counts[i]);
            ++failures;
        }
    }
    if (failures) {
        std::fprintf(stderr, "%d failures\n", failures);
        return 1;
    }
    std::puts("test-affinity: PASS");
    return 0;
}
