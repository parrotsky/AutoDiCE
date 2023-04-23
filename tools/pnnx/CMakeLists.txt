# set cross-compiled system type, it's better not use the type which cmake cannot recognized.
SET ( CMAKE_SYSTEM_NAME Linux )
SET ( CMAKE_SYSTEM_PROCESSOR x86 )
# if gcc/g++ was installed:
SET ( CMAKE_C_COMPILER "gcc" )
SET ( CMAKE_CXX_COMPILER "g++" )

# set searching rules
SET ( CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER )
SET ( CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY )
SET ( CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY )

# set ${CMAKE_C_FLAGS} and ${CMAKE_CXX_FLAGS}flag
SET(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}" )
set(CMAKE_PREFIX_PATH "/home/jq/libtorch")
set(CUDA_TOOLKIT_INCLUDE "/usr/local/cuda")
#SET (CUDA_INCLUDE_DIRS "/usr/local/cuda/include")
SET (CUDA_CUDART_LIBRARY "/usr/local/cuda/lib64")
set(USE_CUDNN 1)
#set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
#set(Protobuf_PREFIX_PATH
#    "/home/jq/protobuf/src"
#    "/home/jq/protobuf"
#    "/usr/local/bin" )
#list(APPEND CMAKE_PREFIX_PATH "${Protobuf_PREFIX_PATH}")
set(CUDA_CUDA_LIBRARY "/usr/local/cuda/lib64/stubs/libcuda.a")



project(pnnx)




if(NOT CMAKE_VERSION VERSION_LESS "3.15")
    # enable CMAKE_MSVC_RUNTIME_LIBRARY
    cmake_policy(SET CMP0091 NEW)
endif()
#set(CMAKE_C_COMPILER gcc-7)
#set(CMAKE_CXX_COMPILER g++-7)
cmake_minimum_required(VERSION 3.12)

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()

if(MSVC AND NOT CMAKE_VERSION VERSION_LESS "3.15")
    option(PNNX_BUILD_WITH_STATIC_CRT "Enables use of statically linked CRT for statically linked pnnx" OFF)
    if(PNNX_BUILD_WITH_STATIC_CRT)
        # cmake before version 3.15 not work
        set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(PNNXPyTorch)

# c++14 is required for using torch headers
set(CMAKE_CXX_STANDARD 14)

#set(CMAKE_BUILD_TYPE debug)
#set(CMAKE_BUILD_TYPE relwithdebinfo)
#set(CMAKE_BUILD_TYPE release)

option(PNNX_COVERAGE "build for coverage" OFF)

#set(Torch_INSTALL_DIR "/home/nihui/.local/lib/python3.9/site-packages/torch" CACHE STRING "")
#set(Torch_INSTALL_DIR "/home/nihui/osd/pnnx/pytorch-v1.10.0/build/install" CACHE STRING "")
#set(Torch_INSTALL_DIR "/home/nihui/osd/pnnx/libtorch" CACHE STRING "")
set(TorchVision_INSTALL_DIR "/home/nihui/osd/vision/build/install" CACHE STRING "")

#set(Torch_DIR "${Torch_INSTALL_DIR}/share/cmake/Torch")
set(TorchVision_DIR "${TorchVision_INSTALL_DIR}/share/cmake/TorchVision")

find_package(Python3 COMPONENTS Interpreter Development)

PNNXProbeForPyTorchInstall()
find_package(Torch REQUIRED)

find_package(TorchVision QUIET)

message(STATUS "Torch_VERSION = ${Torch_VERSION}")
message(STATUS "Torch_VERSION_MAJOR = ${Torch_VERSION_MAJOR}")
message(STATUS "Torch_VERSION_MINOR = ${Torch_VERSION_MINOR}")
message(STATUS "Torch_VERSION_PATCH = ${Torch_VERSION_PATCH}")

if(Torch_VERSION VERSION_LESS "1.8")
    message(FATAL_ERROR "pnnx only supports PyTorch >= 1.8")
endif()

if(TorchVision_FOUND)
    message(STATUS "Building with TorchVision")
    add_definitions(-DPNNX_TORCHVISION)
else()
    message(WARNING "Building without TorchVision")
endif()

include_directories(${TORCH_INCLUDE_DIRS})

add_subdirectory(src)

enable_testing()
add_subdirectory(tests)
