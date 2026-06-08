#include <md/interactions/LJPotential_CLL.cuh>

#include <thrust/sort.h>
#include <thrust/transform_reduce.h>
#include <cub/cub.cuh>

#include <md/core/State.cuh>
#include <md/utils/NeighbourList_CLL.cuh>


using CubicCell = md::cells::CubicCell;

namespace {
    __host__ __device__ __forceinline__ float LJpotential(const float rij1, const float sij1) {
        const float rij2 = rij1 * rij1;
        const float rij6 = rij2 * rij2 * rij2;
        const float sij2 = sij1 * sij1;
        const float sij6 = sij2 * sij2 * sij2;

        return 4.0f * sij6 * (sij6 - rij6) / (rij6 * rij6);
    }

    __host__ __device__ __forceinline__ float deriv_1st_LJpotential(const float rij1, const float sij1) {
        const float rij2 = rij1 * rij1;
        const float rij6 = rij2 * rij2 * rij2;
        const float sij2 = sij1 * sij1;
        const float sij6 = sij2 * sij2 * sij2;

        return -24.0f / rij1 * sij6 * (2.0f * sij6 - rij6) / (rij6 * rij6);
    }

    __global__ void calc_force_kernel(
        int num_species, 
        int num_atoms, 
        int max_neighbours, 
        dfloat3 pos, 
        dfloat3 force, 
        lj_params params, 
        const unsigned int* __restrict__ list, 
        const unsigned int* __restrict__ count, 
        void (*apply_pbc_ptr) (float*, float*, float*, float*), 
        float* lattice
    ) {
        int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
        int warp_id = global_tid / 32;
        int lane_id = global_tid % 32;

        int i = warp_id;

        if (i >= num_atoms) return;
        // スレッド毎の力
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        const float pxi = pos.x[i];
        const float pyi = pos.y[i];
        const float pzi = pos.z[i];
        const int si = params.identifier[i];
        const int num_neighbours = count[i];

        // 1つのワープで一つの粒子の隣接リスト内の粒子から受ける力を計算
        for (int c = lane_id; c < num_neighbours; c += 32) {
            int j = list[i * max_neighbours + c];
            if (j == i) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];
            const int sj = params.identifier[j];
    
            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
        
            apply_pbc_ptr(&dx, &dy, &dz, lattice);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            const float rc = params.cutoff[si * num_species + sj];   
            
            if (dist_sq < rc * rc) {
                const float rij1_inv = rsqrtf(dist_sq);
                const float rij1 = dist_sq * rij1_inv;
                const float sij1 = params.sigma[si * num_species + sj];

                float deriv_1st = deriv_1st_LJpotential(rij1, sij1) - params.deriv_1st_LJpotential_cutoff[si * num_species + sj];
                deriv_1st *= params.epsilon[si * num_species + sj];

                fx += -deriv_1st * dx * rij1_inv;
                fy += -deriv_1st * dy * rij1_inv;
                fz += -deriv_1st * dz * rij1_inv;
            }
        }

        unsigned int mask = 0xffffffff;
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            fx += __shfl_down_sync(mask, fx, offset);
            fy += __shfl_down_sync(mask, fy, offset);
            fz += __shfl_down_sync(mask, fz, offset);
        }

        if (lane_id == 0) {
            force.x[i] = fx;
            force.y[i] = fy;
            force.z[i] = fz;        
        }
    }

    struct CalcPotential {
        const dfloat3 pos;
        const lj_params params;
        int num_species;
        int max_neighbours;
        const unsigned int* __restrict__ list;
        const unsigned int* __restrict__ count;
        void (*apply_pbc_ptr) (float*, float*, float*, float*); 
        float* lattice; 

        CalcPotential(
            dfloat3 _pos, 
            lj_params _params,  
            int _num_species, 
            int _max_neighbours,
            const unsigned int* _list,
            const unsigned int* _count,
            void (*_apply_pbc_ptr) (float*, float*, float*, float*), 
            float* _lattice
        ) : 
        pos(_pos), 
        params(_params),
        num_species(_num_species), 
        max_neighbours(_max_neighbours),
        list(_list),
        count(_count),
        apply_pbc_ptr(_apply_pbc_ptr), 
        lattice(_lattice) {}

        __device__ float operator() (const int i) const {
            float potential = 0.0f;

            const auto pxi = pos.x[i];
            const auto pyi = pos.y[i];
            const auto pzi = pos.z[i];
            const int si = params.identifier[i];
            const int num_neigh = count[i];

            for (int c = 0; c < num_neigh; c++) {
                int j = list[i * max_neighbours + c];

                if (j <= i) continue;

                const auto pxj = pos.x[j];
                const auto pyj = pos.y[j];
                const auto pzj = pos.z[j];
                const int sj = params.identifier[j];

                auto dx = pxi - pxj;
                auto dy = pyi - pyj;
                auto dz = pzi - pzj;

                apply_pbc_ptr(&dx, &dy, &dz, lattice);

                const auto dist_sq = dx * dx + dy * dy + dz * dz;
                const auto r_c = params.cutoff[si * num_species + sj];

                if (dist_sq < r_c * r_c) {
                    const auto rij1 = sqrtf(dist_sq);
                    const auto sij1 = params.sigma[si * num_species + sj];

                    float p = LJpotential(rij1, sij1) - LJpotential(r_c, sij1) - params.deriv_1st_LJpotential_cutoff[si * num_species + sj] * (rij1 - r_c);
                    p *= params.epsilon[si * num_species + sj];

                    potential += p;
                }
            }
            return potential;
        }
    };

    __global__ void apply_sort_kernel(
        const dfloat3 old_force, 
        dfloat3 new_force, 
        const unsigned int* sorted_id, 
        const int num_atoms
    ) {
        unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto old_idx = sorted_id[idx];

        new_force.x[old_idx] = old_force.x[idx];
        new_force.y[old_idx] = old_force.y[idx];
        new_force.z[old_idx] = old_force.z[idx];
    }

    __global__ void apply_forward_sort_kernel(
        const int* __restrict__ original, 
        int* __restrict__ sorted, 
        const unsigned int* __restrict__ sorted_id, 
        const int num_atoms
    ) {
        unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto old_idx = sorted_id[idx];
        sorted[idx] = original[old_idx];
    }
}

using namespace md::interactions;

LJPotential_CLL::LJPotential_CLL(
    int _num_atoms, 
    int _num_species, 
    CubicCell& _cell, 
    NeighbourList_CLL* _nl, 
    std::vector<float> _sigma, 
    std::vector<float> _epsilon, 
    std::vector<float> _cutoff, 
    std::vector<int> _identifier
) : num_species(_num_species), cell(_cell), nl(_nl) {
    // メモリの確保
    size_t size = num_species * num_species;
    size_t N = _num_atoms;
    cudaMalloc(&params.sigma, size * sizeof(float));
    cudaMalloc(&params.epsilon, size * sizeof(float));
    cudaMalloc(&params.cutoff, size * sizeof(float));
    cudaMalloc(&params.deriv_1st_LJpotential_cutoff, size * sizeof(float));
    cudaMalloc(&params.identifier, N * sizeof(int));
    cudaMalloc(&original_identifier, N * sizeof(int));

    cudaMalloc(&force_buffer.x, _num_atoms * sizeof(float));
    cudaMalloc(&force_buffer.y, _num_atoms * sizeof(float));
    cudaMalloc(&force_buffer.z, _num_atoms * sizeof(float));

    // データの転送
    cudaMemcpy(params.sigma, _sigma.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.epsilon, _epsilon.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.cutoff, _cutoff.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(original_identifier, _identifier.data(), N * sizeof(int), cudaMemcpyHostToDevice);

    // カットオフ距離によるLJポテンシャルの一階微分を事前に計算
    std::vector<float> h_deriv_1st_LJpotential_cutoff(size);
    for (int si = 0; si < num_species; si ++) {
        for (int sj = 0; sj < num_species; sj ++) {
            const auto sij1 = _sigma[si * num_species + sj];
            const auto rc = _cutoff[si * num_species + sj];
            h_deriv_1st_LJpotential_cutoff[si * num_species + sj] = deriv_1st_LJpotential(rc, sij1);
        }
    }

    // 転送
    cudaMemcpy(params.deriv_1st_LJpotential_cutoff, h_deriv_1st_LJpotential_cutoff.data(), size * sizeof(float), cudaMemcpyHostToDevice);

    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &calc_force_num_threads, calc_force_kernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &apply_sort_num_threads, apply_sort_kernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &apply_forward_sort_num_threads, apply_forward_sort_kernel, 0, 0);
}

LJPotential_CLL::~LJPotential_CLL() {
    cudaFree(params.sigma);
    cudaFree(params.epsilon);
    cudaFree(params.cutoff);
    cudaFree(params.identifier);
    cudaFree(original_identifier);
    cudaFree(params.deriv_1st_LJpotential_cutoff);
    cudaFree(force_buffer.x);
    cudaFree(force_buffer.y);
    cudaFree(force_buffer.z);
}

void LJPotential_CLL::calc_force(State& state) {
    nl->check(state, cell);
    auto N = state.n_atoms;

    int grid_size_forward_sort = (N + apply_forward_sort_num_threads - 1) / apply_forward_sort_num_threads;
    int grid_size_sort = (N + apply_sort_num_threads - 1) / apply_sort_num_threads;
    int warps_per_block = calc_force_num_threads / 32;
    int calc_force_grid_size = (N + warps_per_block - 1) / warps_per_block;

    auto& cll = nl->get_cell_list();
    auto pid = cll.get_particle_id();
    dfloat3 sorted_pos = cll.get_sorted_pos();

    apply_forward_sort_kernel<<<grid_size_forward_sort, apply_forward_sort_num_threads, 0, state.stream>>>(
        original_identifier, 
        params.identifier, 
        pid, 
        N
    );

    calc_force_kernel<<<calc_force_grid_size, calc_force_num_threads, 0, state.stream>>>(
        num_species, 
        N, 
        nl->get_max_neighbours(), 
        cll.get_sorted_pos(), 
        force_buffer,  
        params, 
        nl->get_list(), 
        nl->get_count(), 
        cell.apply_pbc_ptr, 
        cell.d_lattice
    );

    apply_sort_kernel<<<grid_size_sort, apply_sort_num_threads, 0, state.stream>>>(
        force_buffer, 
        state.force, 
        pid, 
        N
    );
}

void LJPotential_CLL::calc_potential(State& state) {
    nl->check(state, cell);

    auto N = state.n_atoms;

    constexpr int block_size = 256;
    int grid_size_sort = (N + block_size - 1) / block_size;
    auto& cll = nl->get_cell_list();
    auto pid = cll.get_particle_id();

    apply_forward_sort_kernel<<<grid_size_sort, block_size, 0, state.stream>>>(
        original_identifier, 
        params.identifier, 
        pid, 
        N
    );

    // ポテンシャルの計算
    state.potential_energy = thrust::transform_reduce(
        thrust::cuda::par.on(state.stream), 
        thrust::make_counting_iterator<size_t>(0), 
        thrust::make_counting_iterator<size_t>(N), 
        CalcPotential(
            cll.get_sorted_pos(), 
            params, 
            num_species, 
            nl->get_max_neighbours(),
            nl->get_list(),
            nl->get_count(),
            cell.apply_pbc_ptr, 
            cell.d_lattice
        ), 
        0.0f, 
        thrust::plus<float>()
    );
}