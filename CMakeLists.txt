cmake_minimum_required(VERSION 3.10)
project(MD_MLP)

set(CMAKE_CUDA_ARCHITECTURES native)

# C++17 を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Python3 COMPONENTS Development REQUIRED)

add_definitions(-U_FORTIFY_SOURCE)
add_definitions(-D_FORTIFY_SOURCE=0)

# include ディレクトリをインクルードパスに追加
include_directories(${PROJECT_SOURCE_DIR}/include)

# libtorchの場所
set(LIBTORCH_PATH "/home/nozawa/miniconda3/envs/GNN/lib/python3.13/site-packages/torch")
set(Torch_DIR "${LIBTORCH_PATH}/share/cmake/Torch")
find_package(Torch REQUIRED PATHS ${Torch_DIR})

# onnxruntimeのパス
set(ONNXRUNTIME_ROOT "/home/nozawa/onnxruntime-linux-x64-gpu-1.26.0")
include_directories(${ONNXRUNTIME_ROOT}/include)
link_directories(${ONNXRUNTIME_ROOT}/lib)

# torch-trtのパス
set(TORCH_TRT_PATH "/home/nozawa/miniconda3/envs/GNN/lib/python3.13/site-packages/torch_tensorrt")
include_directories(${TORCH_TRT_PATH}/include)
link_directories(${TORCH_TRT_PATH}/lib)

# ソースファイルのリスト
file(GLOB_RECURSE MD_SOURCES 
    "src/md/*.cpp" 
    "src/md/*.cu"
    "/home/nozawa/SchNet/CUDA/*.cu"
    "/home/nozawa/SchNet/CUDA/*.cpp"
)

# 実行ファイルを作成
add_executable(MD_MLP
    src/main.cu
    ${MD_SOURCES}
)

target_include_directories(MD_MLP PRIVATE 
    ${PROJECT_SOURCE_DIR}/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${CUDNN_INCLUDE_DIR}
    ${TORCH_TRT_PATH}/include
)

# libtorch + cuDNN をリンク
target_link_libraries(MD_MLP 
    curand 
    onnxruntime
    ${Python3_LIBRARIES} 
    ${TORCH_LIBRARIES} 
    ${CUDNN_LIBRARY}
    torchtrt
)

target_compile_options(MD_MLP PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        -Xcompiler=-O3,-fPIC
    >
)

set_target_properties(MD_MLP PROPERTIES CUDA_ARCHITECTURES 89 CUDA_SEPARABLE_COMPILATION ON)

set_property(TARGET MD_MLP PROPERTY POSITION_INDEPENDENT_CODE ON)

# cmake -G Ninja -B build -D CMAKE_CUDA_COMPILER=/usr/local/cuda-13/bin/nvcc -D TensorRT_ROOT=/home/nozawa/miniconda3/envs/GNN/lib/python3.13/site-packages/tensorrt_libs .
# ninja -C build