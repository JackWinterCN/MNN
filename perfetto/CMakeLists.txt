file(GLOB MNN_PERFETTO_SRCS
    ${CMAKE_CURRENT_LIST_DIR}/sdk/*.cc
    ${CMAKE_CURRENT_LIST_DIR}/sdk/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/sdk/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
find_package(Threads)

message(STATUS "MNN perfetto Depends:" ${MNN_PERFETTO_SRCS})
add_library(
    MNN_PERFETTO
    STATIC
    ${MNN_PERFETTO_SRCS}
)
set_target_properties(MNN_PERFETTO PROPERTIES
    CXX_VISIBILITY_PRESET default  # 强制 perfetto 的所有符号默认可见
)

target_include_directories(MNN_PERFETTO PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/
    ${CMAKE_CURRENT_LIST_DIR}/sdk/
)

target_link_libraries(MNN_PERFETTO PRIVATE ${CMAKE_THREAD_LIBS_INIT})
