#include <md/neighbour/CellListNoGraph.hpp>

#include <md/core/State.hpp>
#include <md/core/constant.h>
#include <md/core/Cell.cuh>
#include <cub/cub.cuh>
#include <thrust/binary_search.h>

using Cell = md::Cell;
using DeviceVec3 = md::DeviceVec3;
using DeviceInt3 = md::DeviceInt3;

namespace {
    __global__ void calc_cell_id_kernel(
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
}

namespace md {
    void CellListNoGraph::generate(State* state, SimState& simstate, Cell& cell, bool* flag) {
        auto N = state->n_atoms;
        int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;

        calc_cell_id_kernel<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            state->pos, 
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

    void CellListNoGraph::sort(State* state, SimState& simstate, bool* flag) {
        auto N = state->n_atoms;

        int* cell_id_ptr = thrust::raw_pointer_cast(cell_id.data());
        int* perm_ptr = thrust::raw_pointer_cast(perm.data());
        int* sorted_cell_id_ptr = thrust::raw_pointer_cast(sorted_cell_id.data());
        int* sorted_perm_ptr = thrust::raw_pointer_cast(sorted_perm.data());
        int* cell_start_idx_ptr = thrust::raw_pointer_cast(cell_start_idx.data());

        // ソート
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
        state->reorder(sorted_perm_ptr, simstate.stream);
        state->swap_buffer();

        // セルの始まり・終わりの位置を取得
        thrust::counting_iterator<int> search_begin(0);
        thrust::lower_bound(
            thrust::cuda::par_nosync.on(simstate.stream), 
            sorted_cell_id.begin(), 
            sorted_cell_id.begin() + N, 
            search_begin, 
            search_begin + num_cells + 1, 
            cell_start_idx.begin()
        );
    }
}