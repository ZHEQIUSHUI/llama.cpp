set(TARGET cosyvoice3-core)
add_library(${TARGET} STATIC cosy-tts.cpp)
target_link_libraries(${TARGET} PUBLIC llama llama-common mtmd ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PUBLIC cxx_std_17)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(GGML_AXERA AND GGML_AXERA_BSP_DIR)
    find_path(COSY_AXERA_INCLUDE_DIR ax_engine_api.h PATHS ${GGML_AXERA_BSP_DIR}/include NO_DEFAULT_PATH REQUIRED)
    find_library(COSY_AXERA_ENGINE_LIB ax_engine PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
    find_library(COSY_AXERA_INTERPRETER_LIB ax_interpreter PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
    find_library(COSY_AXERA_SYS_LIB ax_sys PATHS ${GGML_AXERA_BSP_DIR}/lib NO_DEFAULT_PATH REQUIRED)
    target_include_directories(${TARGET} PUBLIC ${COSY_AXERA_INCLUDE_DIR})
    target_link_libraries(${TARGET} PUBLIC ${COSY_AXERA_ENGINE_LIB} ${COSY_AXERA_INTERPRETER_LIB} ${COSY_AXERA_SYS_LIB})
    target_compile_definitions(${TARGET} PUBLIC COSY_TTS_HAVE_AXERA=1)
else()
    message(STATUS "cosyvoice3-core: AXERA accelerator support disabled")
endif()

# ONNX Runtime remains only for the reference-audio encoders.  The unified
# AX8860 build does not pass ONNXRUNTIME_ROOT, so use the BSP library and the
# headers vendored beside this source file by default.
if(NOT DEFINED ONNXRUNTIME_ROOT AND GGML_AXERA_BSP_DIR)
    set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
    find_library(COSY_ORT_LIB onnxruntime PATHS "${GGML_AXERA_BSP_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
endif()
if(DEFINED ONNXRUNTIME_ROOT)
    target_include_directories(${TARGET} PRIVATE ${ONNXRUNTIME_ROOT}/include)
    if(COSY_ORT_LIB)
        target_link_libraries(${TARGET} PUBLIC ${COSY_ORT_LIB})
    else()
        target_link_directories(${TARGET} PUBLIC ${ONNXRUNTIME_ROOT}/lib)
        target_link_libraries(${TARGET} PUBLIC onnxruntime)
    endif()
else()
    message(FATAL_ERROR "cosyvoice3-core requires -DONNXRUNTIME_ROOT=<onnxruntime sysroot>")
endif()
