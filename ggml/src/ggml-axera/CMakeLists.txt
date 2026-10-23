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
                         ggml-axera-util.cpp
                         ggml-axera-buffer.cpp
                         ggml-axera-weights.cpp
                         ggml-axera-graph.cpp
                         ggml-axera-attn.cpp
                         ggml-axera-q35-bind.cpp
                         ggml-axera-q35-run.cpp
                         ggml-axera-q35-moe.cpp
                         ggml-axera-vit.cpp
                         ggml-axera-audio.cpp
                         ggml-axera-gen.cpp
                        )

target_include_directories(ggml-axera PRIVATE ${AXERA_INCLUDE_DIR})
target_link_libraries(ggml-axera PRIVATE ${AXERA_ENGINE_LIB} ${AXERA_INTERPRETER_LIB} ${AXERA_SYS_LIB})
# the BSP libraries depend on further board libraries (/opt/lib) that are only resolved at run time
target_link_options(ggml-axera PUBLIC -Wl,--allow-shlib-undefined)
