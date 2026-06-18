#include <md/interactions/NNP_CSR.cuh>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cub/cub.cuh>

#include <md/core/State.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/cells/Cell.cuh>

namespace {
    __global__ void count_pairs_kernel(
        dfloat3 pos, 
        int* __restrict__ counts, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        void (*apply_pbc_ptr) (float*, float*, float*, float*), 
        float* lattice, 
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

            if (idx == j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];
            
            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
        
            apply_pbc_ptr(&dx, &dy, &dz, lattice);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                valid_pairs++;
            }
        }
        counts[idx] = valid_pairs;
    }

    __global__ void build_graph_kernel(
        dfloat3 pos, 
        int32_t* __restrict__ edge_index_ptr, 
        float* __restrict__ edge_weight_ptr, 
        const int32_t* __restrict__ offsets, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        void (*apply_pbc_ptr) (float*, float*, float*, float*), 
        float* lattice, 
        const float cutoff, 
        const int num_atoms, 
        const int max_neighbours, 
        const int num_edges
    ) {
        const int idx = threadIdx.x + blockIdx.x * blockDim.x;
        if (idx >= num_atoms) return;

        const float pxi = pos.x[idx];
        const float pyi = pos.y[idx];
        const float pzi = pos.z[idx];

        int write_idx = offsets[idx];

        for (int c = 0; c < count[idx]; c ++) {
            int j = list[idx * max_neighbours + c];
            if (idx == j) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];

            float dx, dy, dz;
            // 常にインデックスが小さい方を基準に計算
            if (idx < j) {
                dx = pxj - pxi;
                dy = pyj - pyi;
                dz = pzj - pzi;
                apply_pbc_ptr(&dx, &dy, &dz, lattice);
            } else {
                dx = pxi - pxj;
                dy = pyi - pyj;
                dz = pzi - pzj;
                apply_pbc_ptr(&dx, &dy, &dz, lattice);
                // 向きを合わせるために反転
                dx = -dx;
                dy = -dy;
                dz = -dz;
            }
            
            const float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq < cutoff * cutoff) {
                edge_index_ptr[write_idx] = idx;
                edge_index_ptr[num_edges + write_idx] = j;

                edge_weight_ptr[write_idx] = dx;
                edge_weight_ptr[num_edges + write_idx] = dy;
                edge_weight_ptr[2 * num_edges + write_idx] = dz;

                write_idx ++;
            }
        }
    }

    __global__ void append_total_sum_kernel(
        const int* src_array, 
        int32_t* dst_array, 
        int N
    ) {
        dst_array[N] = dst_array[N - 1] + (int64_t)src_array[N - 1];
    }
}

using namespace md::interactions;

NNP_CSR::NNP_CSR(
    State& state, 
    Cell* _cell, 
    NeighbourList* _nl, 
    float _cutoff, 
    int _num_max_edges, 
    const std::string model_path
) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges), loader(model_path) {
    auto N = state.n_atoms;

    // メモリの確保
    cudaMalloc(&x_ptr, N * sizeof(int64_t));
    cudaMalloc(&edge_index_ptr, 2 * num_max_edges * sizeof(int32_t));
    cudaMalloc(&edge_weight_ptr, 3 * num_max_edges * sizeof(float));
    cudaMalloc(&offsets_ptr, (N + 1) * sizeof(int32_t));
    cudaMalloc(&counts, N * sizeof(int));

    // 原子番号はシミュレーションを通して変わらないため、最初に初期化する
    // int32_t -> int64_t
    thrust::copy(
        thrust::device, 
        state.atomic_numbers, 
        state.atomic_numbers + N, 
        x_ptr
    );

    // cubのバッファを確保
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, 
        temp_storage_bytes, 
        counts, 
        offsets_ptr, 
        N, 
        state.stream
    );
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

NNP_CSR::~NNP_CSR() {
    cudaFree(x_ptr);
    cudaFree(edge_index_ptr);
    cudaFree(offsets_ptr);
    cudaFree(edge_weight_ptr);
    cudaFree(counts);
    cudaFree(d_temp_storage);
}

void NNP_CSR::create_graph(State& state) {
    int N = state.n_atoms;

    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;

    count_pairs_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        state.pos, 
        counts, 
        nl->get_list(), 
        nl->get_count(), 
        cell->apply_pbc_ptr, 
        cell->d_lattice, 
        cutoff, 
        N, 
        nl->get_max_neighbours()
    );

    // 手前のインデックスまでを加算
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage, 
        temp_storage_bytes, 
        counts, 
        offsets_ptr, 
        N, 
        state.stream
    );
    // offsets_ptr[N]にnum_edgesを書き込む
    append_total_sum_kernel<<<1, 1, 0, state.stream>>>(
        counts, 
        offsets_ptr, 
        N
    );

    int h_num_edges = 0;
    cudaMemcpyAsync(&h_num_edges, offsets_ptr + N, sizeof(int), cudaMemcpyDeviceToHost, state.stream);
    cudaStreamSynchronize(state.stream);
    this->num_edges = h_num_edges;

    build_graph_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        state.pos, 
        edge_index_ptr, 
        edge_weight_ptr, 
        offsets_ptr, 
        nl->get_list(), 
        nl->get_count(), 
        cell->apply_pbc_ptr, 
        cell->d_lattice, 
        cutoff, 
        N, 
        nl->get_max_neighbours(), 
        num_edges
    );
}

void NNP_CSR::calc_force(State& state) {
    int N = state.n_atoms;
    nl->check(state, cell);
    create_graph(state);

    auto opt = torch::TensorOptions().device(torch::kCUDA);
    inputs = {
        torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt32)), 
        torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32)), 
        torch::from_blob(offsets_ptr, {N + 1}, opt.dtype(torch::kInt32))
    };

    c10::InferenceMode mode;
    int current_device;
    cudaGetDevice(&current_device);
    c10::cuda::CUDAStreamGuard guard(
        c10::cuda::getStreamFromExternal(state.stream, current_device)
    );
    auto outputs = loader.run(inputs);

    float* force_ptr = outputs[1].data_ptr<float>();

    cudaMemcpyAsync(state.force.x, force_ptr, 3 * N * sizeof(float), cudaMemcpyDeviceToDevice, state.stream);
}

void NNP_CSR::calc_potential(State& state) {
    nl->check(state, cell);
    create_graph(state);

    int N = state.n_atoms;

    auto opt = torch::TensorOptions().device(torch::kCUDA);
    inputs = {
        torch::from_blob(x_ptr, {N}, opt.dtype(torch::kInt64)), 
        torch::from_blob(edge_index_ptr, {2, num_edges}, opt.dtype(torch::kInt32)), 
        torch::from_blob(edge_weight_ptr, {3, num_edges}, opt.dtype(torch::kFloat32)), 
        torch::from_blob(offsets_ptr, {N + 1}, opt.dtype(torch::kInt32))
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