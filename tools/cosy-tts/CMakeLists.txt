set(TARGET cosy-tts)
add_executable(${TARGET} cosy-tts.cpp)
target_link_libraries(${TARGET} PRIVATE llama llama-common mtmd ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_include_directories(${TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/ggml/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../mtmd)

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
