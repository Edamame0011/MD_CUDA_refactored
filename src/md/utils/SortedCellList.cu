#include <md/utils/SortedCellList.cuh>

#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>

#include <cub/cub.cuh>
#include <thrust/binary_search.h>

namespace {
    __global__ void calc_cell_id_kernel(
        bool* flag, 
        const dfloat3 pos, 
        int* __restrict__ cell_id, 
        int* __restrict__ particle_id, 
        const int num_atoms, 
        const int Mx, 
        const int My, 
        const int Mz, 
        float* lattice, 
        const float cell_size_inv_x, 
        const float cell_size_inv_y, 
        const float cell_size_inv_z
    ) {
        if (!*flag) return;

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto px = pos.x[idx];
        auto py = pos.y[idx];
        auto pz = pos.z[idx];

        // pbc補正
        px -= lattice[0 * 3 + 0] * floorf(px / lattice[0 * 3 + 0]);
        py -= lattice[1 * 3 + 1] * floorf(py / lattice[1 * 3 + 1]);
        pz -= lattice[2 * 3 + 2] * floorf(pz / lattice[2 * 3 + 2]);

        // セルインデックスの計算
        auto cx = max(0, min(Mx - 1, (int)(px * cell_size_inv_x)));
        auto cy = max(0, min(My - 1, (int)(py * cell_size_inv_y)));
        auto cz = max(0, min(Mz - 1, (int)(pz * cell_size_inv_z)));
        auto icell = cx + cy * Mx + cz * Mx * My;

        cell_id[idx] = icell;
        particle_id[idx] = idx;
    }

    __global__ void apply_sort_kernel(
        const dfloat3 pos, 
        dfloat3 sorted_pos, 
        const int* sorted_particle_id, 
        const int num_atoms
    ) {
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto old_idx = sorted_particle_id[idx];

        sorted_pos.x[idx] = pos.x[old_idx];
        sorted_pos.y[idx] = pos.y[old_idx];
        sorted_pos.z[idx] = pos.z[old_idx];
    }

    __global__ void check_and_sort_kernel(
        bool* flag, 
        void* d_temp_storage, 
        size_t temp_storage_bytes, 
        int* cell_id, 
        int* sorted_cell_id, 
        int* particle_id, 
        int* sorted_particle_id, 
        int num_atoms
    ) {
        if (*flag) {
            cub::DeviceRadixSort::SortPairs(
                d_temp_storage, 
                temp_storage_bytes, 
                cell_id, 
                sorted_cell_id, 
                particle_id, 
                sorted_particle_id, 
                num_atoms
            );
        }
    }
}

using namespace md;

SortedCellList::SortedCellList(std::array<int, 3> _M, Cell* _cell, State& state) : M(_M), cell(_cell) {
    const auto N = state.n_atoms;
    const std::array<std::array<float, 3>, 3>& lattice = cell->lattice;

    num_cells = M[0] * M[1] * M[2];
    for (size_t i = 0; i < 3; i ++) {
        cell_size[i] = (float)(lattice[i][i] / M[i]);
    }

    cudaMalloc(&cell_id, N * sizeof(int));
    cudaMalloc(&particle_id, N * sizeof(int));
    cudaMalloc(&sorted_cell_id, N * sizeof(int));
    cudaMalloc(&sorted_particle_id, N * sizeof(int));
    cudaMalloc(&cell_start_idx, (num_cells + 1) * sizeof(int));
    cudaMalloc(&sorted_pos.x, N * sizeof(float));
    cudaMalloc(&sorted_pos.y, N * sizeof(float));
    cudaMalloc(&sorted_pos.z, N * sizeof(float));

    // cubのバッファを確保
    cub::DeviceRadixSort::SortPairs(
        d_temp_storage, 
        temp_storage_bytes, 
        cell_id, 
        sorted_cell_id, 
        particle_id, 
        sorted_particle_id, 
        N
    );

    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // 最適なスレッド数を計算
    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &calc_cell_id_num_threads, calc_cell_id_kernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &apply_sort_num_threads, apply_sort_kernel, 0, 0);
}

SortedCellList::~SortedCellList() {
    cudaFree(cell_id);
    cudaFree(particle_id);
    cudaFree(cell_start_idx);
    cudaFree(sorted_cell_id);
    cudaFree(sorted_particle_id);
    cudaFree(sorted_pos.x);
    cudaFree(sorted_pos.y);
    cudaFree(sorted_pos.z);
    cudaFree(d_temp_storage);
}

void SortedCellList::generate(State& state, bool* flag) {
    auto N = state.n_atoms;

    int num_blocks = (N + calc_cell_id_num_threads - 1) / calc_cell_id_num_threads;

    calc_cell_id_kernel<<<num_blocks, calc_cell_id_num_threads, 0, state.stream>>>(
        flag, 
        state.pos, 
        cell_id, 
        particle_id, 
        N, 
        M[0], 
        M[1], 
        M[2], 
        cell->d_lattice, 
        1.0f / cell_size[0], 
        1.0f / cell_size[1], 
        1.0f / cell_size[2]
    );
}

void SortedCellList::sort(State& state, bool* flag) {
    auto N = state.n_atoms;

    // ソート
    int num_blocks = (N + apply_sort_num_threads - 1) / apply_sort_num_threads;

    check_and_sort_kernel<<<1, 1, 0, state.stream>>>(
        flag, 
        d_temp_storage, 
        temp_storage_bytes, 
        cell_id, 
        sorted_cell_id, 
        particle_id, 
        sorted_particle_id, 
        N
    ); 

    apply_sort_kernel<<<num_blocks, apply_sort_num_threads, 0, state.stream>>>(
        state.pos, 
        sorted_pos, 
        sorted_particle_id, 
        N
    );

    // セルの始まり・終わりの位置を取得
    thrust::counting_iterator<int> search_begin(0);
    thrust::lower_bound(
        thrust::cuda::par_nosync.on(state.stream), 
        sorted_cell_id, 
        sorted_cell_id + N, 
        search_begin, 
        search_begin + num_cells + 1, 
        cell_start_idx
    );
}