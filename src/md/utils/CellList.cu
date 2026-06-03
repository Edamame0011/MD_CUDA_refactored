#include <md/utils/CellList.cuh>
#include <cub/cub.cuh>
#include <thrust/binary_search.h>

namespace {
    __global__ void calc_cell_id_kernel(
        bool* flag, 
        const dfloat3 pos, 
        unsigned int* __restrict__ cell_id, 
        unsigned int* __restrict__ particle_id, 
        const int num_atoms, 
        const int M, 
        const float Lbox, 
        const float Linv, 
        const float cell_size_inv
    ) {
        if (!*flag) return;

        unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto px = pos.x[idx];
        auto py = pos.y[idx];
        auto pz = pos.z[idx];

        // pbc補正
        px -= Lbox * floorf(px * Linv);
        py -= Lbox * floorf(py * Linv);
        pz -= Lbox * floorf(pz * Linv);

        // セルインデックスの計算
        auto cx = max(0, min(M - 1, (int)(px * cell_size_inv)));
        auto cy = max(0, min(M - 1, (int)(py * cell_size_inv)));
        auto cz = max(0, min(M - 1, (int)(pz * cell_size_inv)));
        auto icell = cx + cy * M + cz * M * M;

        cell_id[idx] = icell;
        particle_id[idx] = idx;
    }

    __global__ void apply_sort_kernel(
        const dfloat3 pos, 
        dfloat3 sorted_pos, 
        const unsigned int* sorted_particle_id, 
        const int num_atoms
    ) {
        unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
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
        unsigned int* cell_id, 
        unsigned int* sorted_cell_id, 
        unsigned int* particle_id, 
        unsigned int* sorted_particle_id, 
        unsigned int num_atoms
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

CellList::CellList(int _M, float _Lbox, State& state) : M(_M), Lbox(_Lbox) {
    const auto N = state.n_atoms;
    const int num_cells = M * M * M;
    cell_size = (float)(Lbox / M);

    cudaMalloc(&cell_id, N * sizeof(unsigned int));
    cudaMalloc(&particle_id, N * sizeof(unsigned int));
    cudaMalloc(&sorted_cell_id, N * sizeof(unsigned int));
    cudaMalloc(&sorted_particle_id, N * sizeof(unsigned int));
    cudaMalloc(&cell_start_idx, (num_cells + 1) * sizeof(unsigned int));
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

CellList::~CellList() {
    cudaFree(cell_id);
    cudaFree(particle_id);
    cudaFree(cell_start_idx);
    cudaFree(sorted_cell_id);
    cudaFree(sorted_particle_id);
    cudaFree(sorted_pos.x);
    cudaFree(sorted_pos.y);
    cudaFree(sorted_pos.z);
}

void CellList::generate(State& state, bool* flag) {
    auto N = state.n_atoms;

    int num_blocks = (N + calc_cell_id_num_threads - 1) / calc_cell_id_num_threads;

    calc_cell_id_kernel<<<num_blocks, calc_cell_id_num_threads, 0, state.stream>>>(
        flag, 
        state.pos, 
        cell_id, 
        particle_id, 
        N, 
        M, 
        Lbox, 
        1.0f / Lbox, 
        1.0f / cell_size
    );
}

void CellList::sort(State& state, bool* flag) {
    auto N = state.n_atoms;
    const int num_cells = M * M * M;

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
    thrust::counting_iterator<unsigned int> search_begin(0);
    thrust::lower_bound(
        thrust::cuda::par_nosync.on(state.stream), 
        sorted_cell_id, 
        sorted_cell_id + N, 
        search_begin, 
        search_begin + num_cells + 1, 
        cell_start_idx
    );
}