#include <md/interactions/NNP_aoti.cuh>
#include <md/cells/CubicCell.cuh>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch_tensorrt/torch_tensorrt.h>

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

            float dx = pxj - pxi;
            float dy = pyj - pyi;
            float dz = pzj - pzi;
        
            cell.apply_pbc(dx, dy, dz);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                // i -> j
                edge_index_ptr[write_idx] = idx;
                edge_index_ptr[num_edges + write_idx] = j;

                edge_weight_ptr[write_idx] = dx;
                edge_weight_ptr[num_edges + write_idx] = dy;
                edge_weight_ptr[2 * num_edges + write_idx] = dz;

                // j -> iへコピー
                int rev_idx = write_idx + num_pairs;
                edge_index_ptr[rev_idx] = j;
                edge_index_ptr[num_edges + rev_idx] = idx;

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
NNP_aoti<CellType>::NNP_aoti(
    State& state, 
    CellType _cell, 
    md::utils::NeighbourList<CellType>* _nl, 
    float _cutoff, 
    int _num_max_edges, 
    const std::string model_path
) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges), loader(model_path) {
    auto N = state.n_atoms;
    auto view = state.get_view();

    // メモリの確保
    cudaMalloc(&x_ptr, N * sizeof(int64_t));
    cudaMalloc(&edge_weight_ptr, 3 * num_max_edges * sizeof(float));
    cudaMalloc(&edge_index_ptr, 2 * num_max_edges * sizeof(int64_t));
    cudaMalloc(&counts, N * sizeof(int));
    cudaMalloc(&offsets, N * sizeof(int));

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
NNP_aoti<CellType>::~NNP_aoti() {
    cudaFree(x_ptr);
    cudaFree(edge_weight_ptr);
    cudaFree(edge_index_ptr);
    cudaFree(counts);
    cudaFree(offsets);
}

template <typename CellType>
void NNP_aoti<CellType>::create_graph(State& state) {
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
void NNP_aoti<CellType>::calc_force(State& state) {
    int N = state.n_atoms;
    nl->check(state, cell);
    create_graph(state);
    auto view = state.get_view();

    auto opt = torch::TensorOptions().device(torch::kCUDA);
    inputs = {
        torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32))
    };

    c10::InferenceMode mode;
    int current_device;
    cudaGetDevice(&current_device);
    c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(state.stream, current_device)
    );
    auto outputs = loader.run(inputs);

    float* force_ptr = outputs[1].data_ptr<float>();

    cudaMemcpyAsync(view.force.x, force_ptr, 3 * N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
}

template <typename CellType>
void NNP_aoti<CellType>::calc_potential(State& state) {
    nl->check(state, cell);
    create_graph(state);

    int N = state.n_atoms;
    auto view = state.get_view();

    auto opt = torch::TensorOptions().device(torch::kCUDA);
    inputs = {
        torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32))
    };

    c10::InferenceMode mode;
    int current_device;
    cudaGetDevice(&current_device);
    c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(state.stream, current_device)
    );

    auto outputs = loader.run(inputs);

    float* energy_ptr = outputs[0].data_ptr<float>();

    cudaMemcpy(&state.potential_energy, energy_ptr, sizeof(float), cudaMemcpyDeviceToHost);
}

template class NNP_aoti<md::cells::CubicCell>;