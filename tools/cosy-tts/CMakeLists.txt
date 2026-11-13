set(TARGET cosy-tts)
add_executable(${TARGET} cosy-tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd)

if(GGML_AXERA AND GGML_AXERA_BSP_DIR)
    find_path(COSY_AXERA_INCLUDE_DIR ax_engine_api.h PATHS ${GGML_AXERA_BSP_DIR}/include NO_DEFAULT_PATH REQUIRED)
    find_library(COSY_AXERA_ENGINE_LIB ax_engine PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
    find_library(COSY_AXERA_INTERPRETER_LIB ax_interpreter PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
    find_library(COSY_AXERA_SYS_LIB ax_sys PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
    target_include_directories(${TARGET} PRIVATE ${COSY_AXERA_INCLUDE_DIR})
    target_link_libraries(${TARGET} PRIVATE ${COSY_AXERA_ENGINE_LIB} ${COSY_AXERA_INTERPRETER_LIB} ${COSY_AXERA_SYS_LIB})
    target_compile_definitions(${TARGET} PRIVATE COSY_TTS_HAVE_AXERA=1)
else()
    message(STATUS "cosy-tts: AXERA accelerator support disabled; flow/HiFT run through ONNX Runtime")
endif()

# ONNX Runtime is used only by this tool.  Set ONNXRUNTIME_ROOT to a sysroot
# with include/onnxruntime_cxx_api.h and lib/libonnxruntime.so.
if(DEFINED ONNXRUNTIME_ROOT)
    target_include_directories(${TARGET} PRIVATE ${ONNXRUNTIME_ROOT}/include)
    target_link_directories(${TARGET} PRIVATE ${ONNXRUNTIME_ROOT}/lib)
    target_link_libraries(${TARGET} PRIVATE onnxruntime)
else()
    message(FATAL_ERROR "cosy-tts requires -DONNXRUNTIME_ROOT=<onnxruntime sysroot>")
endif()

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
