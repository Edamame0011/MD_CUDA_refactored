#include <md/interactions/NNP.hpp>

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include <md/core/State.hpp>
#include <md/neighbour/NeighbourList.hpp>
#include <md/core/Cell.cuh>

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>

using DeviceVec3 = md::DeviceVec3;
using Cell = md::Cell;

namespace {
    __global__ void count_pairs_kernel(
        DeviceVec3 pos, 
        int* __restrict__ counts, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        Cell cell, 
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
        
            cell.apply_pbc_device(&dx, &dy, &dz);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            
            if (dist_sq < cutoff * cutoff) {
                valid_pairs++;
            }
        }
        counts[idx] = valid_pairs;
    }

    __global__ void build_graph_kernel(
        DeviceVec3 pos, 
        int64_t* __restrict__ edge_index_ptr, 
        float* __restrict__ edge_weight_ptr, 
        const int* __restrict__ offsets,
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        Cell cell, 
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
        
            cell.apply_pbc_device(&dx, &dy, &dz);
    
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

namespace md::interactions {
    NNP::NNP(
        State* state, 
        Cell& _cell, 
        NeighbourList* _nl, 
        float _cutoff, 
        int _num_max_edges, 
        const std::string model_path
    ) : cell(_cell), cutoff(_cutoff), nl(_nl), num_max_edges(_num_max_edges), loader(model_path) {
        auto N = state->n_atoms;

        // メモリの確保
        cudaMalloc(&x_ptr, N * sizeof(int64_t));
        cudaMalloc(&edge_weight_ptr, 3 * num_max_edges * sizeof(float));
        cudaMalloc(&edge_index_ptr, 2 * num_max_edges * sizeof(int64_t));
        cudaMalloc(&counts, N * sizeof(int));
        cudaMalloc(&offsets, N * sizeof(int));
    }

    NNP::~NNP() {
        cudaFree(x_ptr);
        cudaFree(edge_weight_ptr);
        cudaFree(edge_index_ptr);
        cudaFree(counts);
        cudaFree(offsets);
    }

    void NNP::create_graph(State* state, SimState& simstate) {
        int N = state->n_atoms;

        int num_threads = 256;
        int num_blocks = (N + num_threads - 1) / num_threads;

        count_pairs_kernel<<<num_blocks, num_threads, 0, simstate.stream>>>(
            state->pos, 
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
            thrust::cuda::par.on(simstate.stream),
            counts,
            counts + N,
            offsets
        );

        int last_count, last_offset;
        cudaMemcpyAsync(&last_count, counts + N - 1, sizeof(int), cudaMemcpyDeviceToHost, simstate.stream);
        cudaMemcpyAsync(&last_offset, offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost, simstate.stream);

        cudaStreamSynchronize(simstate.stream);

        int num_pairs = last_offset + last_count;
        num_edges = 2 * num_pairs;

        if (num_edges > num_max_edges) {
            throw std::runtime_error("NNP: エッジ数が最大エッジ数を超過しました。");
        }

        build_graph_kernel<<<num_blocks, num_threads, 0, simstate.stream>>>(
            state->pos, 
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

        // 元素番号のコピー
        // int32_t -> int64_t
        thrust::copy(
            thrust::device, 
            state->atomic_number, 
            state->atomic_number + N, 
            x_ptr
        );
    }

    void NNP::calc_force(State* state, SimState& simstate) {
        int N = state->n_atoms;

        nl->check(state, simstate, cell);
        create_graph(state, simstate);

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
            c10::cuda::getStreamFromExternal(simstate.stream, current_device)
        );
        auto outputs = loader.run(inputs);

        float* force_ptr = outputs[1].data_ptr<float>();

        cudaMemcpyAsync(state->force.x, force_ptr, 3 * N * sizeof(float), cudaMemcpyDeviceToDevice, simstate.stream);
    }

    float NNP::calc_potential(State* state, SimState& simstate) {
        nl->check(state, simstate, cell);
        create_graph(state, simstate);

        int N = state->n_atoms;

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
            c10::cuda::getStreamFromExternal(simstate.stream, current_device)
        );

        auto outputs = loader.run(inputs);

        float* energy_ptr = outputs[0].data_ptr<float>();
        float p;
        cudaMemcpy(&p, energy_ptr, sizeof(float), cudaMemcpyDeviceToHost);

        return p;
    }
}