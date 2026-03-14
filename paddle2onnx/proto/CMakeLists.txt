include_directories(${CMAKE_CURRENT_SOURCE_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_library(p2o_paddle_proto p2o_paddle.proto)
target_link_libraries(p2o_paddle_proto PUBLIC protobuf::libprotobuf)
protobuf_generate(TARGET p2o_paddle_proto)
