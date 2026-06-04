#include <md/interactions/LJPotential.cuh>

#include <thrust/transform_reduce.h>
#include <cub/cub.cuh>

#include <md/core/State.cuh>
#include <md/cells/Cell.cuh>
#include <md/utils/NeighbourList.cuh>

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
        const int* __restrict__ list, 
        const int* __restrict__ count, 
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
        const int* __restrict__ list;
        const int* __restrict__ count;
        void (*apply_pbc_ptr) (float*, float*, float*, float*);
        float* lattice;

        CalcPotential(
            dfloat3 _pos, 
            lj_params _params,  
            int _num_species, 
            int _max_neighbours,
            const int* _list,
            const int* _count,
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

        __device__ float operator() (int i) {
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

                    auto p = LJpotential(rij1, sij1) - LJpotential(r_c, sij1) - params.deriv_1st_LJpotential_cutoff[si * num_species + sj] * (rij1 - r_c);
                    p *= params.epsilon[si * num_species + sj];

                    potential += p;
                }
            }
            return potential;
        }
    };
}

using namespace md::interactions;

LJPotential::LJPotential(
    int _num_species, 
    Cell* _cell, 
    NeighbourList *_nl, 
    std::vector<float> _sigma, 
    std::vector<float> _epsilon, 
    std::vector<float> _cutoff, 
    std::vector<int> _identifier
) : num_species(_num_species), cell(_cell), nl(_nl) {
    // メモリの確保
    size_t size = num_species * num_species;
    size_t N = _identifier.size();
    cudaMalloc(&params.sigma, size * sizeof(float));
    cudaMalloc(&params.epsilon, size * sizeof(float));
    cudaMalloc(&params.cutoff, size * sizeof(float));
    cudaMalloc(&params.deriv_1st_LJpotential_cutoff, size * sizeof(float));
    cudaMalloc(&params.identifier, N * sizeof(int));

    // データの転送
    cudaMemcpy(params.sigma, _sigma.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.epsilon, _epsilon.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.cutoff, _cutoff.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(params.identifier, _identifier.data(), N * sizeof(int), cudaMemcpyHostToDevice);

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
}

LJPotential::~LJPotential() {
    cudaFree(params.sigma);
    cudaFree(params.epsilon);
    cudaFree(params.cutoff);
    cudaFree(params.identifier);
    cudaFree(params.deriv_1st_LJpotential_cutoff);
}

void LJPotential::calc_force(State& state) {
    nl->check(state, cell);
    auto N = state.n_atoms;

    int warps_per_block = calc_force_num_threads / 32;

    int grid_size = (N + warps_per_block - 1) / warps_per_block;

    calc_force_kernel<<<grid_size, calc_force_num_threads, 0, state.stream>>>(
        num_species, 
        N, 
        nl->get_max_neighbours(), 
        state.pos, 
        state.force, 
        params, 
        nl->get_list(), 
        nl->get_count(), 
        cell->apply_pbc_ptr, 
        cell->d_lattice
    );
}

void LJPotential::calc_potential(State& state) {
    auto N = state.n_atoms;

    // ポテンシャルの計算
    state.potential_energy = thrust::transform_reduce(
        thrust::cuda::par.on(state.stream), 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        CalcPotential(
            state.pos, 
            params, 
            num_species, 
            nl->get_max_neighbours(),
            nl->get_list(),
            nl->get_count(),
            cell->apply_pbc_ptr, 
            cell->d_lattice
        ), 
        0.0f, 
        thrust::plus<float>()
    );
}