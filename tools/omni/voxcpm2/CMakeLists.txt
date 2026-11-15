set(CORE_TARGET voxcpm2-core)
add_library(${CORE_TARGET} STATIC
    voxcpm2_audiovae.cpp
    voxcpm2_cfm.cpp
    voxcpm2_components.cpp
    voxcpm2_fsq.cpp
    voxcpm2_llm.cpp
    voxcpm2_localenc.cpp
    voxcpm2_locdit.cpp
    voxcpm2_runtime.cpp
    voxcpm2_transformer.cpp
    voxcpm2_wav.cpp
)
target_include_directories(${CORE_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/common)
target_include_directories(${CORE_TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/ggml/src)
target_link_libraries(${CORE_TARGET} PUBLIC llama llama-common ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${CORE_TARGET} PUBLIC cxx_std_17)
if (GGML_AXERA)
    target_compile_definitions(${CORE_TARGET} PRIVATE VOXCPM2_HAVE_AXERA=1)
    target_link_libraries(${CORE_TARGET} PUBLIC ggml-axera)
endif()

set(TARGET voxcpm2-cli)
add_executable(${TARGET} voxcpm2_cli.cpp)

target_link_libraries(${TARGET} PRIVATE ${CORE_TARGET})
target_compile_features(${TARGET} PRIVATE cxx_std_17)

if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
