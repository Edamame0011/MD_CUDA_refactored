#include <md/interactions/NNP_onnx.cuh>
#include <md/cells/CubicCell.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <thrust/execution_policy.h>

namespace {
    template <typename CellType>
    __global__ void count_pairs_kernel(
        dfloat3 pos, 
        int* __restrict__ counts, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        CellType cell, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int valid_pairs = 0;
        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];

            if (idx >= j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];
            
            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
        
            cell.apply_pbc(dx, dy, dz);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                valid_pairs++;
            }
        }
        counts[idx] = valid_pairs;
    }

    template <typename CellType>
    __global__ void build_graph_kernel(
        dfloat3 pos, 
        int64_t* __restrict__ edge_index_ptr, 
        float* __restrict__ edge_weight_ptr, 
        const int* __restrict__ offsets,
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        CellType cell, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours, 
        const int num_edges,
        const int num_pairs
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int write_idx = offsets[idx];

        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];
            if (idx >= j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];

            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
        
            cell.apply_pbc(dx, dy, dz);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                // i -> j
                edge_index_ptr[write_idx] = j;
                edge_index_ptr[num_edges + write_idx] = idx;

                edge_weight_ptr[write_idx] = dx;
                edge_weight_ptr[num_edges + write_idx] = dy;
                edge_weight_ptr[2 * num_edges + write_idx] = dz;

                // j -> iへコピー
                int rev_idx = write_idx + num_pairs;
                edge_index_ptr[rev_idx] = idx;
                edge_index_ptr[num_edges + rev_idx] = j;

                edge_weight_ptr[rev_idx] = -dx;
                edge_weight_ptr[num_edges + rev_idx] = -dy;
                edge_weight_ptr[2 * num_edges + rev_idx] = -dz;

                write_idx ++;
            }
        }
    }
}

using namespace md::interactions;

template <typename CellType>
NNP_onnx<CellType>::NNP_onnx(
    State& state, 
    CellType _cell, 
    md::utils::NeighbourList<CellType>* _nl, 
    float _cutoff, 
    int _num_max_edges, 
    const std::string model_path
) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges), env(ORT_LOGGING_LEVEL_WARNING, "ONNX_Inference_Class"), cuda_memory_info("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault) {
    // モデルの読み込み
    Ort::SessionOptions session_opt;
    
    OrtCUDAProviderOptions cuda_opt;
    cuda_opt.device_id = 0;

    cuda_opt.has_user_compute_stream = 1;
    cuda_opt.user_compute_stream = state.stream;
    cuda_opt.arena_extend_strategy = 1;

    session_opt.AppendExecutionProvider_CUDA(cuda_opt);
    session_opt.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_opt);
    io_binding = std::make_unique<Ort::IoBinding>(*session);
    std::cout << "モデルを読み込みました：" << model_path << std::endl;

    auto N = state.n_atoms;
    auto view = state.get_view();

    // メモリの確保
    cudaMalloc(&x_ptr, N * sizeof(int64_t));
    cudaMalloc(&edge_weight_ptr, 3 * num_max_edges * sizeof(float));
    cudaMalloc(&edge_index_ptr, 2 * num_max_edges * sizeof(int64_t));
    cudaMalloc(&counts, N * sizeof(int));
    cudaMalloc(&offsets, N * sizeof(int));
    cudaMalloc(&forces_buffer, 3 * N * sizeof(float));
    cudaMalloc(&total_energy_buffer, sizeof(float));

    // 原子番号はシミュレーションを通して変わらないため、最初に初期化する
    // int32_t -> int64_t
    thrust::copy(
        thrust::device, 
        view.atomic_numbers, 
        view.atomic_numbers + N, 
        x_ptr
    );
}

template <typename CellType>
NNP_onnx<CellType>::~NNP_onnx() {
    cudaFree(x_ptr);
    cudaFree(edge_weight_ptr);
    cudaFree(edge_index_ptr);
    cudaFree(counts);
    cudaFree(offsets);
    cudaFree(forces_buffer);
    cudaFree(total_energy_buffer);
}

template <typename CellType>
void NNP_onnx<CellType>::create_graph(State& state) {
    int N = state.n_atoms;
    auto view = state.get_view();

    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;

    count_pairs_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        view.pos, 
        counts, 
        nl->get_list(), 
        nl->get_count(), 
        cell, 
        cutoff, 
        N, 
        nl->get_max_neighbours()
    );

    // 手前のインデックスまでを加算
    thrust::exclusive_scan(
        thrust::cuda::par.on(state.stream),
        counts,
        counts + N,
        offsets
    );

    int last_count, last_offset;
    cudaMemcpyAsync(&last_count, counts + N - 1, sizeof(int), cudaMemcpyDeviceToHost, state.stream);
    cudaMemcpyAsync(&last_offset, offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost, state.stream);

    cudaStreamSynchronize(state.stream);

    int num_pairs = last_offset + last_count;
    num_edges = 2 * num_pairs;

    build_graph_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        view.pos, 
        edge_index_ptr, 
        edge_weight_ptr, 
        offsets, 
        nl->get_list(), 
        nl->get_count(), 
        cell, 
        cutoff, 
        N, 
        nl->get_max_neighbours(), 
        num_edges, 
        num_pairs
    );
}

template <typename CellType>
void NNP_onnx<CellType>::calc_force(State& state) {
    int N = state.n_atoms;
    auto view = state.get_view();
    nl->check(state, cell);
    create_graph(state);

    // ラッパーテンソルの作成
    std::vector<int64_t> x_shape = {N};
    std::vector<int64_t> edge_index_shape = {2, num_edges};
    std::vector<int64_t> edge_weight_shape = {3, num_edges};
    x = Ort::Value::CreateTensor<int64_t>(
        cuda_memory_info, x_ptr, N, x_shape.data(), x_shape.size()
    );
    edge_index = Ort::Value::CreateTensor<int64_t>(
        cuda_memory_info, edge_index_ptr, 2 * num_edges, edge_index_shape.data(), edge_index_shape.size()
    );
    edge_weight = Ort::Value::CreateTensor<float>(
        cuda_memory_info, edge_weight_ptr, 3 * num_edges, edge_weight_shape.data(), edge_weight_shape.size()
    );

    std::vector<int64_t> total_energy_shape = {};
    std::vector<int64_t> forces_shape = {3, N};
    total_energy = Ort::Value::CreateTensor<float>(
        cuda_memory_info, total_energy_buffer, 1, total_energy_shape.data(), total_energy_shape.size()
    );
    forces = Ort::Value::CreateTensor<float>(
        cuda_memory_info, forces_buffer, 3 * N, forces_shape.data(), forces_shape.size()
    );

    // バインディング
    io_binding->ClearBoundInputs();
    io_binding->ClearBoundOutputs();

    io_binding->BindInput("x", x);
    io_binding->BindInput("edge_index", edge_index);
    io_binding->BindInput("edge_weight", edge_weight);

    io_binding->BindOutput("total_energy", total_energy);
    io_binding->BindOutput("forces", forces);

    // 推論
    session->Run(Ort::RunOptions{nullptr}, *io_binding);

    // コピー
    cudaMemcpyAsync(view.force.x, forces_buffer, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
    cudaMemcpyAsync(view.force.y, forces_buffer + N, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
    cudaMemcpyAsync(view.force.z, forces_buffer + N + N, N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
}

template <typename CellType>
void NNP_onnx<CellType>::calc_potential(State& state) {
    int N = state.n_atoms;
    auto view = state.get_view();
    nl->check(state, cell);
    create_graph(state);

    // ラッパーテンソルの作成
    std::vector<int64_t> x_shape = {N};
    std::vector<int64_t> edge_index_shape = {2, num_edges};
    std::vector<int64_t> edge_weight_shape = {3, num_edges};
    x = Ort::Value::CreateTensor<int64_t>(
        cuda_memory_info, x_ptr, N, x_shape.data(), x_shape.size()
    );
    edge_index = Ort::Value::CreateTensor<int64_t>(
        cuda_memory_info, edge_index_ptr, 2 * num_edges, edge_index_shape.data(), edge_index_shape.size()
    );
    edge_weight = Ort::Value::CreateTensor<float>(
        cuda_memory_info, edge_weight_ptr, 3 * num_edges, edge_weight_shape.data(), edge_weight_shape.size()
    );

    std::vector<int64_t> total_energy_shape = {};
    std::vector<int64_t> forces_shape = {3, N};
    total_energy = Ort::Value::CreateTensor<float>(
        cuda_memory_info, total_energy_buffer, 1, total_energy_shape.data(), total_energy_shape.size()
    );
    forces = Ort::Value::CreateTensor<float>(
        cuda_memory_info, forces_buffer, 3 * N, forces_shape.data(), forces_shape.size()
    );

    io_binding->ClearBoundInputs();
    io_binding->ClearBoundOutputs();

    io_binding->BindInput("x", x);
    io_binding->BindInput("edge_index", edge_index);
    io_binding->BindInput("edge_weight", edge_weight);

    io_binding->BindOutput("total_energy", total_energy);
    io_binding->BindOutput("forces", forces);

    // 推論
    session->Run(Ort::RunOptions{nullptr}, *io_binding);

    // コピー
    cudaMemcpyAsync(&state.potential_energy, total_energy_buffer, sizeof(float), cudaMemcpyDeviceToHost, state.stream);
}

template class NNP_onnx<md::cells::CubicCell>;