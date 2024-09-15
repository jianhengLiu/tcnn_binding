cmake_minimum_required(VERSION 3.18)
project(tcnn_binding)

find_package(Torch REQUIRED)

# tiny-cuda-nn
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_definitions(-DJSON_HAS_CPP_14)
set(TCNN_BUILD_BENCHMARK OFF)
set(TCNN_BUILD_EXAMPLES OFF)
add_subdirectory(submodules/tiny-cuda-nn)
include_directories(${TCNN_INCLUDE_DIRECTORIES} submodules/tiny-cuda-nn/include
                    submodules/tiny-cuda-nn/dependencies)

# tcnn_binding
add_library(tcnn_binding tcnn_binding/bindings.cpp
                         tcnn_binding/tcnn_binding.cpp)
target_link_libraries(tcnn_binding ${TORCH_LIBRARIES} tiny-cuda-nn)
