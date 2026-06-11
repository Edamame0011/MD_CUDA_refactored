#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>
#include <md/utils/UnsortedCellList.cuh>

namespace {
    __global__ void build_linked_cell_kernel(
        bool* flag, 
        const dfloat3 pos, 
        int* __restrict__ head, 
        int* __restrict__ next, 
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

        int old_head = atomicExch(&head[icell], idx);
        next[idx] = old_head;
    }
}

using namespace md;

UnsortedCellList::UnsortedCellList(std::array<int, 3> _M, Cell* _cell, State& state) : M(_M), cell(_cell) {
    const auto N = state.n_atoms;
    const std::array<std::array<float, 3>, 3>& lattice = cell->lattice;

    num_cells = M[0] * M[1] * M[2];
    for (size_t i = 0; i < 3; i ++) {
        cell_size[i] = (float)(lattice[i][i] / M[i]);
    }

    // メモリ確保
    cudaMalloc(&head, num_cells * sizeof(int));
    cudaMalloc(&next, N * sizeof(int));
}

UnsortedCellList::~UnsortedCellList() {
    cudaFree(head);
    cudaFree(next);
}

void UnsortedCellList::generate(State& state, bool* flag) {
    auto N = state.n_atoms;

    cudaMemsetAsync(head, 0xFF, num_cells * sizeof(int), state.stream);

    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;

    build_linked_cell_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        flag, 
        state.pos, 
        head, 
        next, 
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