#include <md/interactions/LJPotential.cuh>
#include <md/cells/CubicCell.cuh>

#include <thrust/transform_reduce.h>
#include <cub/cub.cuh>

namespace {
    constexpr int BLOCK_SIZE = 64;

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

    struct Force3 {
        float x, y, z;
    };

    struct SumForce3 {
        __device__ __forceinline__ Force3 operator()(const Force3& a, const Force3& b) const {
            return {a.x + b.x, a.y + b.y, a.z + b.z};
        }
    };

    template <typename CellType>
    __global__ void calc_force_kernel(
        int num_species, 
        int num_atoms, 
        dfloat3 pos, 
        dfloat3 force, 
        lj_params params, 
        const int* __restrict__ list, 
        const int* __restrict__ count, 
        CellType cell
    ) {
        int i = blockIdx.x;
        if (i > num_atoms) return;

        // ブロック内のスレッドのid
        int tid = threadIdx.x;

        // スレッド毎の力
        float fx = 0.0f;
        float fy = 0.0f;
        float fz = 0.0f;

        const float pxi = pos.x[i];
        const float pyi = pos.y[i];
        const float pzi = pos.z[i];
        const int si = params.identifier[i];
        const int num_neighbours = count[i];

        // 1つのブロックで一つの粒子の隣接リスト内の粒子から受ける力を計算
        for (int c = tid; c < num_neighbours; c += blockDim.x) {
            int j = list[i * num_atoms + c];
            if (j == i) continue;

            const float pxj = pos.x[j];
            const float pyj = pos.y[j];
            const float pzj = pos.z[j];
            const int sj = params.identifier[j];
    
            float dx = pxi - pxj;
            float dy = pyi - pyj;
            float dz = pzi - pzj;
    
            cell.apply_pbc(dx, dy, dz);
    
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

        Force3 thread_force = {fx, fy, fz};

        // CUBの共有メモリの確保
        using BlockReduce = cub::BlockReduce<Force3, BLOCK_SIZE>;
        __shared__ typename BlockReduce::TempStorage temp_storage;

        // ブロック内の和をとる
        Force3 block_force = BlockReduce(temp_storage).Reduce(thread_force, SumForce3());

        if (tid == 0) {
            force.x[i] = block_force.x;
            force.y[i] = block_force.y;
            force.z[i] = block_force.z;
        }
    }

    template <typename CellType>
    struct CalcPotential {
        const dfloat3 pos;
        const lj_params params;
        int num_species, num_atoms;
        CellType cell;

        CalcPotential(
            dfloat3 _pos, 
            lj_params _params,  
            int _num_atoms, 
            int _num_species, 
            CellType _cell
        ) : 
        pos(_pos), 
        params(_params),
        num_atoms(_num_atoms), 
        num_species(_num_species), 
        cell(_cell) {}

        __device__ float operator() (int idx) {
            auto i = idx / num_atoms;
            auto j = idx % num_atoms;

            if (j <= i) return 0.0f;

            // 原子の種類
            const auto si = params.identifier[i];
            const auto sj = params.identifier[j];

            // 距離の計算
            const auto x1 = pos.x[i];
            const auto y1 = pos.y[i];
            const auto z1 = pos.z[i];

            const auto x2 = pos.x[j];
            const auto y2 = pos.y[j];
            const auto z2 = pos.z[j];

            auto dx = x1 - x2;
            auto dy = y1 - y2;
            auto dz = z1 - z2;

            cell.apply_pbc(dx, dy, dz);

            const auto dist_sq = dx * dx + dy * dy + dz * dz;

            const auto r_c = params.cutoff[si * num_species + sj];

            // カットオフ距離内かを判定
            if (dist_sq >= r_c * r_c) return 0.0f;

            // ポテンシャルの計算
            const auto rij1 = sqrtf(dist_sq);
            const auto sij1 = params.sigma[si * num_species + sj];

            auto potential = LJpotential(rij1, sij1) - LJpotential(r_c, sij1) - params.deriv_1st_LJpotential_cutoff[si * num_species + sj] * (rij1 - r_c);
            potential *= params.epsilon[si * num_species + sj];

            return potential;
        }
    };
}

using namespace md::interactions;

template <typename CellType>
LJPotential<CellType>::LJPotential(
    int _num_species, 
    CellType _cell, 
    md::utils::NeighbourList<CellType> *_NL, 
    std::vector<float> _sigma, 
    std::vector<float> _epsilon, 
    std::vector<float> _cutoff, 
    std::vector<int> _identifier
) : num_species(_num_species), cell(_cell), NL(_NL) {
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
    for (int i = 0; i < num_species; i ++) {
        const int si = _identifier[i];
        for (int j = 0; j < num_species; j ++) {
            const int sj = _identifier[j];
            const auto sij1 = _sigma[si * num_species + sj];
            const auto rc = _cutoff[si * num_species + sj];
            h_deriv_1st_LJpotential_cutoff[si * num_species + sj] = deriv_1st_LJpotential(rc, sij1);
        }
    }

    // 転送
    cudaMemcpy(params.deriv_1st_LJpotential_cutoff, h_deriv_1st_LJpotential_cutoff.data(), size * sizeof(float), cudaMemcpyHostToDevice);
}

template <typename CellType>
LJPotential<CellType>::~LJPotential() {
    cudaFree(params.sigma);
    cudaFree(params.epsilon);
    cudaFree(params.cutoff);
    cudaFree(params.identifier);
    cudaFree(params.deriv_1st_LJpotential_cutoff);
}

template <typename CellType>
void LJPotential<CellType>::calc_force(State& state) {
    NL->check(state, cell);
    auto N = state.n_atoms;
    auto view = state.get_view();

    int grid_size = N;
    int block_size = BLOCK_SIZE;

    calc_force_kernel<<<grid_size, block_size>>>(
        num_species, 
        N, 
        view.pos, 
        view.force, 
        params, 
        NL->get_list(), 
        NL->get_count(), 
        cell
    );
}

template <typename CellType>
void LJPotential<CellType>::calc_potential(State& state) {
        auto N = state.n_atoms;
        auto view = state.get_view();

        // ポテンシャルの計算
        state.potential_energy = thrust::transform_reduce(
            thrust::device, 
            thrust::make_counting_iterator(0), 
            thrust::make_counting_iterator(N * N), 
            CalcPotential(
                view.pos, 
                params, 
                N, 
                num_species, 
                cell
            ), 
            0.0f, 
            thrust::plus<float>()
        );
}


template class LJPotential<md::cells::CubicCell>;