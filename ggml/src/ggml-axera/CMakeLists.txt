if (NOT GGML_AXERA_BSP_DIR)
    message(FATAL_ERROR "GGML_AXERA requires GGML_AXERA_BSP_DIR (Axera BSP with include/ax_engine_api.h and lib/)")
endif()

find_path(AXERA_INCLUDE_DIR ax_engine_api.h PATHS ${GGML_AXERA_BSP_DIR}/include NO_DEFAULT_PATH REQUIRED)
find_library(AXERA_ENGINE_LIB ax_engine PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(AXERA_INTERPRETER_LIB ax_interpreter PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
find_library(AXERA_SYS_LIB ax_sys PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
message(STATUS "Axera BSP: ${GGML_AXERA_BSP_DIR}")

ggml_add_backend_library(ggml-axera
                         ggml-axera.cpp
                         core/affinity.cpp
                         core/runtime.cpp
                         core/manifest.cpp
                         core/quant.cpp
                         core/params.cpp
                         core/npu_sampling.cpp
                         core/argmax_state.cpp
                         core/seqstate.cpp
                         core/kv_state.cpp
                         core/patch_embed.cpp
                         ops/ops.cpp
                         ops/llm.cpp
                         ops/gdn_bank.cpp
                         ops/gdn_mtp.cpp
                         ops/gdn_state.cpp
                         models/model.cpp
                         models/qwen38_27b/qwen38_27b.cpp
                         models/qwen35_fused/qwen35_fused.cpp
                         models/qwen35_whole/qwen35_whole.cpp
                         models/minicpm_ocr/minicpm_ocr_vit_full.cpp
                         models/minicpmv4_7/minicpmv4_7_vit.cpp
                         models/qwen35_moe_sgl/qwen35_moe_sgl.cpp
                         models/moe_gqa/moe_gqa.cpp
                         models/qwen3/qwen3.cpp
                         models/llama_dense/llama_dense.cpp
                         models/qwen3_asr/qwen3_asr.cpp
                         models/qwen3_tts/qwen3_tts.cpp
                         models/qwen3_tts/codec.cpp
                         models/phase_dense/phase_dense.cpp
                         models/voxcpm2/vox_acoustic.cpp
                         models/qwen35_vit/qwen35_vit.cpp
                         models/qwen35_vit/qwen35_vit_full.cpp
                         models/inkling2/inkling2_vit_full.cpp
                         models/q3vl30/q3vl30_vit_full.cpp
                         models/lightnav/lightnav.cpp
                         models/lightnav/lightnav_vit_full.cpp
                         models/inkling2/inkling2.cpp
                         ggml-axera-buffer.cpp
                        )

target_include_directories(ggml-axera PRIVATE ${AXERA_INCLUDE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../../../vendor)
target_link_libraries(ggml-axera PRIVATE ${AXERA_ENGINE_LIB} ${AXERA_INTERPRETER_LIB} ${AXERA_SYS_LIB})
# the BSP libraries depend on further board libraries (/opt/lib) that are only resolved at run time
target_link_options(ggml-axera PUBLIC -Wl,--allow-shlib-undefined)
