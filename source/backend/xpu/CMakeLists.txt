# Android NN API
file(GLOB MNN_NPU_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/backend/*.hpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/execution/*.hpp
)

add_library(
    MNN_XPU
    OBJECT
    ${MNN_NPU_SRCS}
)

target_include_directories(MNN_XPU PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/backend
    ${CMAKE_CURRENT_LIST_DIR}/execution
)
