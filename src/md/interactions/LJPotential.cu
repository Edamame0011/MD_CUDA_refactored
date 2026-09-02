#include <md/interactions/LJPotential.hpp>

#include <md/core/State.hpp>
#include <md/core/Cell.cuh>
#include <md/core/constant.h>
#include <md/neighbour/NeighbourList.hpp>

#include <thrust/transform_reduce.h>

using DeviceVec3 = md::DeviceVec3;
using Cell = md::Cell;

namespace {
    struct lj_params_view {
        const float *sigma; 
        const float *sigma6; 
        const float *epsilon; 
        const float *cutoff; 
        const float *cutoff_sq; 
        const float *deriv_1st_LJpotential_cutoff;
    };

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
        const int num_species, 
        const int num_atoms, 
        const int max_neighbours, 
        const DeviceVec3 pos, 
        DeviceVec3 force, 
        const int* __restrict__ species, 
        const lj_params_view params, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        const Cell cell
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
        const int si = species[i];
        const int num_neighbours = count[i];

        // 1つのワープで一つの粒子の隣接リスト内の粒子から受ける力を計算
        for (int c = lane_id; c < num_neighbours; c += 32) {
            int j = list[i * max_neighbours + c];
            if (j == i) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];
            const int sj = species[j];
            const int pair_idx = si * num_species + sj;
    
            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
        
            cell.apply_pbc_device(&dx, &dy, &dz);
    
            const float dist_sq = dx * dx + dy * dy + dz * dz;
            const float rc_sq = params.cutoff_sq[pair_idx];   
            
            if (dist_sq < rc_sq) {
                const float rinv = rsqrtf(dist_sq);
                const float s6 = params.sigma6[pair_idx];

                // LJポテンシャルの1階微分を計算 (phi'(r) / r)
                //   - 24 / r^2 [2 * (sigma / r)^12 - (sigma / r)^6]
                // = (sigma / r)^6 * (-48 * (sigma / r)^6 + 24) * (1/r)^2
                const float rinv2 = rinv * rinv;
                const float rinv6 = rinv2 * rinv2 * rinv2;
                const float s6_rinv6 = s6 * rinv6;
                const float deriv_1st_r = s6_rinv6 * __fmaf_rn(-48.0f, s6_rinv6, 24.0f) * rinv2;
                
                // (phi'(r) - phi'(r_c)) * 1/r
                float deriv_1st = deriv_1st_r - params.deriv_1st_LJpotential_cutoff[pair_idx] * rinv;
                deriv_1st *= params.epsilon[pair_idx];

                fx += -deriv_1st * dx;
                fy += -deriv_1st * dy;
                fz += -deriv_1st * dz;
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
        const DeviceVec3 pos;
        const int* __restrict__ species; 
        const lj_params_view params;
        int num_species;
        int max_neighbours;
        const int* __restrict__ list;
        const int* __restrict__ count;
        Cell cell;

        CalcPotential(
            DeviceVec3 _pos, 
            const int* _species, 
            lj_params_view _params,  
            int _num_species, 
            int _max_neighbours,
            const int* _list,
            const int* _count,
            Cell _cell
        ) : 
        pos(_pos), 
        species(_species), 
        params(_params),
        num_species(_num_species), 
        max_neighbours(_max_neighbours),
        list(_list),
        count(_count),
        cell(_cell) {}

        __device__ float operator() (int i) {
            float potential = 0.0f;

            const auto pxi = pos.x[i];
            const auto pyi = pos.y[i];
            const auto pzi = pos.z[i];
            const int si = species[i];
            const int num_neigh = count[i];

            for (int c = 0; c < num_neigh; c++) {
                int j = list[i * max_neighbours + c];

                if (j <= i) continue;

                const auto pxj = pos.x[j];
                const auto pyj = pos.y[j];
                const auto pzj = pos.z[j];
                const int sj = species[j];

                auto dx = pxi - pxj;
                auto dy = pyi - pyj;
                auto dz = pzi - pzj;

                cell.apply_pbc_device(&dx, &dy, &dz);

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

namespace md::interactions {
    LJPotential::LJPotential(
        int _num_species, 
        Cell& cell_, 
        NeighbourList *nl_, 
        std::vector<float> sigma_, 
        std::vector<float> epsilon_, 
        std::vector<float> cutoff_
    ) : num_species(_num_species), cell(cell_), nl(nl_) {
        size_t size = _num_species * _num_species;

        // sigma^6, cutoff^2を事前に計算
        std::vector<float> h_sigma6(size);
        std::vector<float> h_cutoff_sq(size);
        for (int i = 0; i < size; i ++) {
            auto s1 = sigma_[i];
            auto s2 = s1 * s1;
            h_sigma6[i] = s2 * s2 * s2;

            auto c1 = cutoff_[i];
            h_cutoff_sq[i] = c1 * c1;
        }

        // カットオフ距離によるLJポテンシャルの一階微分を事前に計算
        std::vector<float> h_deriv_1st_LJpotential_cutoff(size);
        for (int si = 0; si < num_species; si ++) {
            for (int sj = 0; sj < num_species; sj ++) {
                const auto sij1 = sigma_[si * num_species + sj];
                const auto rc = cutoff_[si * num_species + sj];
                h_deriv_1st_LJpotential_cutoff[si * num_species + sj] = deriv_1st_LJpotential(rc, sij1);
            }
        }

        // データの転送
        params.sigma =  sigma_;
        params.sigma6 = h_sigma6;
        params.epsilon = epsilon_;
        params.cutoff = cutoff_;
        params.cutoff_sq = h_cutoff_sq;
        params.deriv_1st_LJpotential_cutoff = h_deriv_1st_LJpotential_cutoff;
    }

    void LJPotential::calc_force(State& state, SimState& simstate) {
        nl->check(state, simstate, cell);
        auto N = state.n_atoms;

        int num_warps = NUM_THREADS / 32;
        int num_blocks = (N + num_warps - 1) / num_warps;

        lj_params_view view = {
            thrust::raw_pointer_cast(params.sigma.data()), 
            thrust::raw_pointer_cast(params.sigma6.data()), 
            thrust::raw_pointer_cast(params.epsilon.data()), 
            thrust::raw_pointer_cast(params.cutoff.data()), 
            thrust::raw_pointer_cast(params.cutoff_sq.data()), 
            thrust::raw_pointer_cast(params.deriv_1st_LJpotential_cutoff.data())
        };

        calc_force_kernel<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            num_species, 
            N, 
            nl->get_max_neighbours(), 
            state.pos, 
            state.force, 
            state.species, 
            view, 
            nl->get_list(), 
            nl->get_count(), 
            cell
        );
    }

    float LJPotential::calc_potential(State& state, SimState& simstate){
        nl->check(state, simstate, cell);
        auto N = state.n_atoms;

        lj_params_view view = {
            thrust::raw_pointer_cast(params.sigma.data()), 
            thrust::raw_pointer_cast(params.sigma6.data()), 
            thrust::raw_pointer_cast(params.epsilon.data()), 
            thrust::raw_pointer_cast(params.cutoff.data()), 
            thrust::raw_pointer_cast(params.cutoff_sq.data()), 
            thrust::raw_pointer_cast(params.deriv_1st_LJpotential_cutoff.data())
        };

        // ポテンシャルの計算
        float potential_energy = thrust::transform_reduce(
            thrust::cuda::par.on(simstate.stream), 
            thrust::make_counting_iterator<int>(0), 
            thrust::make_counting_iterator<int>(N), 
            CalcPotential(
                state.pos, 
                state.species, 
                view, 
                num_species, 
                nl->get_max_neighbours(),
                nl->get_list(),
                nl->get_count(),
                cell
            ), 
            0.0f, 
            thrust::plus<float>()
        );

        return potential_energy;
    }
}