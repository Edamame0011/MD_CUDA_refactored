cmake_minimum_required(VERSION 3.10)
project(MD_MLP)

# C++17 を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_definitions(-U_FORTIFY_SOURCE)
add_definitions(-D_FORTIFY_SOURCE=0)

# include ディレクトリをインクルードパスに追加
include_directories(${PROJECT_SOURCE_DIR}/include)

# libtorchの場所
set(LIBTORCH_PATH "/home/nozawa/miniconda3/envs/MD_CUDA/lib/python3.11/site-packages/torch")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# cuDNNの設定
set(CUDNN_ROOT "/home/nozawa/miniconda3/envs/MD_CUDA/") 
find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDNN_ROOT}/include /usr/include)
find_library(CUDNN_LIBRARY cudnn PATHS ${CUDNN_ROOT}/lib64 ${CUDNN_ROOT}/lib /usr/lib/x86_64-linux-gnu)

if (CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
    message(STATUS "Found cuDNN: ${CUDNN_LIBRARY}")
    include_directories(${CUDNN_INCLUDE_DIR})
    add_definitions(-DUSE_CUDNN=1)
else()
    message(WARNING "cuDNN not found. Compiling without cuDNN support.")
    add_definitions(-DUSE_CUDNN=0)
endif()

# ソースファイルのリスト
file(GLOB_RECURSE MD_SOURCES 
    "src/md/*.cpp" 
    "src/md/*.cu"
)

# 実行ファイルを作成
add_executable(MD_MLP
    src/main.cu
    ${MD_SOURCES}
)

# libtorch + cuDNN をリンク
target_link_libraries(MD_MLP ${TORCH_LIBRARIES} ${CUDNN_LIBRARY})

target_compile_options(MD_MLP PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-O3,-fPIC
    >
)

set_target_properties(MD_MLP PROPERTIES CUDA_ARCHITECTURES 89)

set_property(TARGET MD_MLP PROPERTY POSITION_INDEPENDENT_CODE ON)

# cmake -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc ..