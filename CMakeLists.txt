cmake_minimum_required(VERSION 3.18)

# CUDAアーキテクチャの指定
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR
   CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be specified.")
endif()

project(MD_MLP LANGUAGES CXX CUDA)

# C++17 を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(NOT TARGET CUDA::nvToolsExt)
    add_library(CUDA::nvToolsExt INTERFACE IMPORTED)
endif()

# ソースファイルのリスト
file(GLOB_RECURSE MD_SOURCES
    "src/md/*.cpp"
    "src/md/*.cu"
)

# 実行ファイルを作成
add_executable(MD_MLP
    src/main.cpp
    ${MD_SOURCES}
)

target_include_directories(MD_MLP PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(MD_MLP PRIVATE
    ${Python3_LIBRARIES}
    ${TORCH_LIBRARIES}
    CUDA::curand
    util
)

target_compile_options(MD_MLP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-O3,-fPIC
    >
)

set_target_properties(MD_MLP PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)