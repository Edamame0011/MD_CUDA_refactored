#include <md/neighbour/CellList.hpp>

#include <md/core/State.hpp>
#include <md/core/Cell.cuh>
#include <md/core/constant.h>

#include <cub/cub.cuh>

using Cell = md::Cell;
using DeviceVec3 = md::DeviceVec3;

namespace {
    __global__ void calc_cell_id_kernel(
        const bool* flag, 
        const DeviceVec3 pos, 
        int* __restrict__ cell_id, 
        int* __restrict__ perm, 
        const int num_atoms, 
        const int Mx, 
        const int My, 
        const int Mz, 
        Cell cell, 
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
        cell.apply_pbc_wrap_device(&px, &py, &pz);

        // セルインデックスの計算
        int cx = max(0, min(Mx - 1, (int)(px * cell_size_inv_x)));
        int cy = max(0, min(My - 1, (int)(py * cell_size_inv_y)));
        int cz = max(0, min(Mz - 1, (int)(pz * cell_size_inv_z)));
        int icell = cx + cy * Mx + cz * Mx * My;

        cell_id[idx] = icell;
        perm[idx] = idx;
    }

    __global__ void apply_sort_kernel(
        const bool* flag, 
        const int* __restrict__ perm, 
        const DeviceVec3 pos, 
        const DeviceVec3 vel, 
        const float* __restrict__ mass, 
        const float* __restrict__ mass_inv, 
        const int* __restrict__ species, 
        const int* __restrict__ particle_id, 
        DeviceVec3 pos_buffer, 
        DeviceVec3 vel_buffer, 
        float* __restrict__ mass_buffer, 
        float* __restrict__ mass_inv_buffer, 
        int* __restrict__ species_buffer, 
        int* __restrict__ particle_id_buffer, 
        const int num_atoms
    ) {
        if (!*flag) return;

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto old_idx = perm[idx];

        pos_buffer.x[idx] = pos.x[old_idx];
        pos_buffer.y[idx] = pos.y[old_idx];
        pos_buffer.z[idx] = pos.z[old_idx];
        vel_buffer.x[idx] = vel.x[old_idx];
        vel_buffer.y[idx] = vel.y[old_idx];
        vel_buffer.z[idx] = vel.z[old_idx];
        mass_buffer[idx] = mass[old_idx];
        mass_inv_buffer[idx] = mass_inv[old_idx];
        species_buffer[idx] = species[old_idx];
        particle_id_buffer[idx] = particle_id[old_idx];
    }

    __global__ void commit_sort_kernel(
        const bool* flag, 
        DeviceVec3 pos, 
        DeviceVec3 vel, 
        float* __restrict__ mass, 
        float* __restrict__ mass_inv, 
        int* __restrict__ species, 
        int* __restrict__ particle_id, 
        const DeviceVec3 pos_buffer, 
        const DeviceVec3 vel_buffer, 
        const float* __restrict__ mass_buffer, 
        const float* __restrict__ mass_inv_buffer, 
        const int* __restrict__ species_buffer, 
        const int* __restrict__ particle_id_buffer, 
        const int num_atoms
    ) {
        if (!*flag) return;

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        pos.x[idx] = pos_buffer.x[idx];
        pos.y[idx] = pos_buffer.y[idx];
        pos.z[idx] = pos_buffer.z[idx];
        vel.x[idx] = vel_buffer.x[idx];
        vel.y[idx] = vel_buffer.y[idx];
        vel.z[idx] = vel_buffer.z[idx];
        mass[idx] = mass_buffer[idx];
        mass_inv[idx] = mass_inv_buffer[idx];
        species[idx] = species_buffer[idx];
        particle_id[idx] = particle_id_buffer[idx];
    }

    __global__ void calc_cell_start_kernel(
        const bool* flag, 
        const int* sorted_cell_id,
        int* cell_start_idx,
        int N,
        int num_cells
    ) {
        if (!*flag) return;

        int cell = blockIdx.x * blockDim.x + threadIdx.x;

        if (cell > num_cells) return;

        // lower_bound(sorted_cell_id, cell)
        int lo = 0;
        int hi = N;

        while (lo < hi) {
            int mid = (lo + hi) / 2;

            if (sorted_cell_id[mid] < cell)
                lo = mid + 1;
            else
                hi = mid;
        }

        cell_start_idx[cell] = lo;
    }
}

namespace md {
    CellList::CellList(std::array<int, 3> M_, State& state, Cell& cell)
    : M(M_) {
        const auto N = state.n_atoms;
        const auto& lattice = cell.get_lattice();

        num_cells = M[0] * M[1] * M[2];
        for (size_t i = 0; i < 3; i ++) {
            cell_size[i] = (float)(lattice[i] / M[i]);
        }

        cell_id.resize(N);
        perm.resize(N);
        sorted_cell_id.resize(N);
        sorted_perm.resize(N);
        cell_start_idx.resize(num_cells + 1);

        // cubのバッファを確保
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, 
            temp_storage_bytes, 
            thrust::raw_pointer_cast(cell_id.data()), 
            thrust::raw_pointer_cast(sorted_cell_id.data()), 
            thrust::raw_pointer_cast(perm.data()), 
            thrust::raw_pointer_cast(sorted_perm.data()), 
            N
        );

        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    CellList::~CellList() {
        cudaFree(d_temp_storage);
    }

    void CellList::generate(State& state, SimState& simstate, Cell& cell, bool* flag) {
        auto N = state.n_atoms;
        int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;

        calc_cell_id_kernel<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            thrust::raw_pointer_cast(cell_id.data()), 
            thrust::raw_pointer_cast(perm.data()), 
            N, 
            M[0], 
            M[1], 
            M[2], 
            cell, 
            1.0f / cell_size[0], 
            1.0f / cell_size[1], 
            1.0f / cell_size[2]
        );
    }

    void CellList::sort(State& state, SimState& simstate, bool* flag) {
        auto N = state.n_atoms;

        int* cell_id_ptr = thrust::raw_pointer_cast(cell_id.data());
        int* perm_ptr = thrust::raw_pointer_cast(perm.data());
        int* sorted_cell_id_ptr = thrust::raw_pointer_cast(sorted_cell_id.data());
        int* sorted_perm_ptr = thrust::raw_pointer_cast(sorted_perm.data());
        int* cell_start_idx_ptr = thrust::raw_pointer_cast(cell_start_idx.data());

        // ソート
        int num_blocks_sort = (N + NUM_THREADS - 1) / NUM_THREADS;
        cub::DeviceRadixSort::SortPairs(
            d_temp_storage, 
            temp_storage_bytes, 
            cell_id_ptr, 
            sorted_cell_id_ptr, 
            perm_ptr, 
            sorted_perm_ptr,  
            N,
            0,
            sizeof(int) * 8,
            simstate.stream
        );

        apply_sort_kernel<<<num_blocks_sort, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            sorted_perm_ptr, 
            state.pos, 
            state.vel, 
            state.mass, 
            state.mass_inv, 
            state.species, 
            state.particle_id, 
            state.pos_buffer, 
            state.vel_buffer, 
            state.mass_buffer, 
            state.mass_inv_buffer, 
            state.species_buffer, 
            state.particle_id_buffer, 
            N
        );

        // 元配列の書き換え（cudaGraphs対応のためswapではなく書き換える）
        commit_sort_kernel<<<num_blocks_sort, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            state.vel, 
            state.mass, 
            state.mass_inv, 
            state.species, 
            state.particle_id, 
            state.pos_buffer, 
            state.vel_buffer, 
            state.mass_buffer, 
            state.mass_inv_buffer, 
            state.species_buffer, 
            state.particle_id_buffer, 
            N
        );

        // セルの始まり・終わりの位置を取得
        int num_blocks_cell = (num_cells + NUM_THREADS) / NUM_THREADS;
        calc_cell_start_kernel<<<num_blocks_cell, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            sorted_cell_id_ptr, 
            cell_start_idx_ptr, 
            N, 
            num_cells
        );
    }
}