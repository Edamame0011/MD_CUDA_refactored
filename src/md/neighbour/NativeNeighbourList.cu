#include <md/neighbour/NativeNeighbourList.hpp>

#include <md/core/Cell.cuh>
#include <md/core/constant.h>
#include <md/neighbour/NL_utils.cuh>
#include <md/neighbour/CellList.hpp>

using Top2 = md::neighbour::Top2;
using Cell = md::Cell;
using DeviceVec3 = md::DeviceVec3;

constexpr int WARP_SIZE = 32;

namespace {
    __global__ void generate_nl_kernel(
        bool* flag, 
        const DeviceVec3 pos, 
        const DeviceVec3 nl_conf, 
        const int* __restrict__ species, 
        const int num_atoms, 
        const int num_species, 
        const int max_neighbours, 
        int* __restrict__ list, 
        int* __restrict__ count, 
        const float* __restrict__  cutoff_margin_sq, 
        Cell cell
    ) {
        if (!*flag) return;

        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        int warp_id = tid / 32;
        int lane_id = tid % 32;

        int i = warp_id;

        if (i < num_atoms) {

            float pxi, pyi, pzi;
            int si;

            if (lane_id == 0) {
                pxi = pos.x[i];
                pyi = pos.y[i];
                pzi = pos.z[i];
                si = species[i];
            }

            pxi = __shfl_sync(0xffffffff, pxi, 0);
            pyi = __shfl_sync(0xffffffff, pyi, 0);
            pzi = __shfl_sync(0xffffffff, pzi, 0);
            si = __shfl_sync(0xffffffff, si, 0);

            int c = 0;

            for (int j = 0; j < num_atoms; j += 32) {
                int j_curr = j + lane_id;
                bool is_neighbour = false;

                if (j_curr < num_atoms && i != j_curr) {
                    auto pxj = pos.x[j_curr];
                    auto pyj = pos.y[j_curr];
                    auto pzj = pos.z[j_curr];
                    int sj = species[j_curr];

                    // 距離の計算
                    auto dx = pxi - pxj;
                    auto dy = pyi - pyj;
                    auto dz = pzi - pzj;
                    
                    cell.apply_pbc_device(&dx, &dy, &dz);

                    const auto dist_sq = dx * dx + dy * dy + dz * dz;

                    if (dist_sq < cutoff_margin_sq[si * num_species + sj]) {
                        is_neighbour = true;
                    }
                }
            
                unsigned int mask = __ballot_sync(0xffffffff, is_neighbour);

                if (is_neighbour) {
                    int offset = __popc(mask & ((1u << lane_id) - 1));
                    int dst = c + offset;
                    if (dst < max_neighbours) {
                        list[i * max_neighbours + dst] = j_curr;
                    }
                }

                c += __popc(mask);
            }

            if (lane_id == 0) {
                count[i] = min(c, max_neighbours);
                nl_conf.x[i] = pxi;
                nl_conf.y[i] = pyi;
                nl_conf.z[i] = pzi;
            }
        }
    }
}

namespace md::neighbour {
    NativeNeighbourList::NativeNeighbourList(int n_atoms, int max_neighbours_, std::vector<float> cutoff_, float margin_) 
    : NeighbourList(n_atoms, max_neighbours_, margin_) {
        cudaMalloc(&this->flag, sizeof(bool));
        cudaMemset(this->flag, 1, sizeof(bool));

        NeighbourList::init_cutoff(cutoff_);
    }

    NativeNeighbourList::~NativeNeighbourList() {
        cudaFree(d_temp_storage);
        cudaFree(this->flag);
    }

    void NativeNeighbourList::generate(State& state, SimState& simstate, Cell& cell) {
        auto N = state.n_atoms;

        // nlの作成
        int num_warps = NUM_THREADS / WARP_SIZE;
        int num_blocks = (N + num_warps - 1) / num_warps;
        
        generate_nl_kernel<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            nl_conf, 
            state.species, 
            N, 
            num_species, 
            max_neighbours, 
            thrust::raw_pointer_cast(list.data()), 
            thrust::raw_pointer_cast(count.data()), 
            thrust::raw_pointer_cast(cutoff_margin_sq.data()), 
            cell
        );

        cudaMemsetAsync(this->flag, 0, sizeof(bool), simstate.stream);

        // バッファの確保
        CalcDist op(
            state.pos, 
            this->nl_conf, 
            cell
        );
        thrust::counting_iterator<int> count_itr(0);
        auto trans_itr = thrust::make_transform_iterator(count_itr, op);

        cub::DeviceReduce::Reduce(
            this->d_temp_storage, 
            this->temp_storage_bytes, 
            trans_itr, 
            this->top2, 
            N, 
            MergeTop2(), 
            Top2()
        );

        cudaMallocAsync(&d_temp_storage, temp_storage_bytes, simstate.stream);
    }

    void NativeNeighbourList::check(State& state, SimState& simstate, Cell& cell) {
        auto N = state.n_atoms;

        // 移動距離の大きい順に2粒子の移動距離を表すTop2オブジェクトを計算
        CalcDist op(
            state.pos, 
            this->nl_conf, 
            cell
        );
        thrust::counting_iterator<int> count_itr(0);
        auto trans_itr = thrust::make_transform_iterator(count_itr, op);

        cub::DeviceReduce::Reduce(
            this->d_temp_storage, 
            this->temp_storage_bytes, 
            trans_itr, 
            this->top2, 
            N, 
            MergeTop2(), 
            Top2(), 
            simstate.stream
        );

        // Top2オブジェクトの移動距離が閾値を超えているか判定し、超えていたらGenerate()を呼ぶ
        check_top2<<<1, 1, 0, simstate.stream>>>(
            this->top2, 
            this->flag, 
            margin * margin
        );

        constexpr int num_warps = NUM_THREADS / WARP_SIZE;
        int generate_nl_num_blocks = (N + num_warps - 1) / num_warps;
        int update_nl_conf_num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;

        generate_nl_kernel<<<generate_nl_num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            nl_conf, 
            state.species, 
            N, 
            num_species, 
            max_neighbours, 
            thrust::raw_pointer_cast(list.data()), 
            thrust::raw_pointer_cast(count.data()), 
            thrust::raw_pointer_cast(cutoff_margin_sq.data()), 
            cell
        );

        update_nl_conf_kernel<<<update_nl_conf_num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            nl_conf, 
            N
        );

        cudaMemsetAsync(this->flag, 0, sizeof(bool), simstate.stream);
    }
}
