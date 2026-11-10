// Host-side wall-clock profiling for board experiments.
//
// LLAMA_AXERA_HOST_PROF=1 is an umbrella switch: llama_backend_init() maps it to the existing
// GGML_AXERA_PROF and LLAMA_GRAPH_PROF switches, and a few llama.cpp-side stages that the
// backend cannot see (state streaming, sampling) add aggregate timers here.  With the switch
// absent the checks compile to one cached branch and nothing is printed.
#pragma once

#include "ggml.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>

inline bool llama_host_prof_on() {
    static const bool on = [] {
        const char * e = std::getenv("LLAMA_AXERA_HOST_PROF");
        return e && std::atoi(e) != 0;
    }();
    return on;
}

struct llama_host_prof_acc {
    struct entry {
        double  ms = 0.0;
        double  min = 1.0e300;
        double  max = 0.0;
        int64_t n  = 0;
    };
    std::map<std::string, entry> t;
    std::mutex                   mu;

    ~llama_host_prof_acc() {
        for (const auto & [name, v] : t) {
            std::fprintf(stderr, "host prof: %-34s %10.2f ms %8lld calls %8.3f ms/call min %8.3f max %8.3f\n",
                         name.c_str(), v.ms, (long long) v.n, v.ms / (double) (v.n ? v.n : 1), v.min, v.max);
        }
    }
};

inline llama_host_prof_acc & llama_host_prof() {
    static llama_host_prof_acc acc;
    return acc;
}

struct llama_host_prof_scope {
    const char *                               name;
    std::chrono::steady_clock::time_point      t0;

    explicit llama_host_prof_scope(const char * n) : name(llama_host_prof_on() ? n : nullptr) {
        if (name) {
            t0 = std::chrono::steady_clock::now();
        }
    }
    ~llama_host_prof_scope() {
        if (!name) {
            return;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        auto & acc = llama_host_prof();
        std::lock_guard<std::mutex> lock(acc.mu);
        auto & e = acc.t[name];
        e.ms += ms;
        e.min = std::min(e.min, ms);
        e.max = std::max(e.max, ms);
        ++e.n;
    }
    llama_host_prof_scope(const llama_host_prof_scope &) = delete;
    llama_host_prof_scope & operator=(const llama_host_prof_scope &) = delete;
};
