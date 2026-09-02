#include <md/neighbour/CellListNeighbourList.hpp>

#include <md/core/Cell.cuh>
#include <md/core/constant.h>
#include <md/neighbour/NL_utils.cuh>
#include <md/neighbour/CellList.hpp>

#include <cooperative_groups.h>

using Top2 = md::neighbour::Top2;
using Cell = md::Cell;
using DeviceVec3 = md::DeviceVec3;

namespace cg = cooperative_groups;

// 取り敢えず8スレッド毎に並列化（他の数字の方が良いかは未検証）
constexpr int TILE_SIZE = 8;

namespace {
    __global__ void generate_nl_kernel(
        const bool* __restrict__ flag, 
        const DeviceVec3 pos, 
        const int num_atoms, 
        const int max_neighbours, 
        const int Mx, 
        const int My, 
        const int Mz, 
        int* __restrict__ list, 
        int* __restrict__ count, 
        const int* __restrict__ cell_id, 
        const int* __restrict__ cell_start_idx, 
        const float cutoff_margin_sq, 
        Cell cell
    ) { 
        if (!*flag) return;

        cg::thread_block block = cg::this_thread_block();
        auto tile = cg::tiled_partition<TILE_SIZE>(block);

        const int lane_id = tile.thread_rank();
        const int tile_in_block = threadIdx.x / TILE_SIZE;
        const int tiles_per_block = blockDim.x / TILE_SIZE;

        int idx = blockIdx.x * tiles_per_block + tile_in_block;

        if (idx >= num_atoms) return;

        int cid;
        float pxi, pyi, pzi;

        if (lane_id == 0) {
            auto cid = cell_id[idx];
            auto pxi = pos.x[idx];
            auto pyi = pos.y[idx];
            auto pzi = pos.z[idx];
        }

        cid = tile.shfl(cid, 0);
        pxi = tile.shfl(pxi, 0);
        pyi = tile.shfl(pyi, 0);
        pzi = tile.shfl(pzi, 0);

        int c = 0;

        // 粒子iのセルインデックス
        int cx = cid % Mx;
        int cy  = (cid / Mx) % My;
        int cz = cid / (Mx * My);

        for (int dz = -1; dz <= 1; dz ++) {
            // 粒子jのセルインデックス
            int jz = cz + dz;
            if (jz < 0)
                jz += Mz;
            else if (jz >= Mz)
                jz -= Mz;

            for (int dy = -1; dy <= 1; dy ++) {
                int jy = cy + dy;
                if (jy < 0)
                    jy += My;
                else if (jy >= My)
                    jy -= My;

                for (int dx = -1; dx <= 1; dx ++) {
                    int jx = cx + dx;
                    if (jx < 0)
                        jx += Mx;
                    else if (jx >= Mx)
                        jx -= Mx;
                    
                    // 粒子jが所属するセル
                    int jcell = jx + jy * Mx + jz * Mx * My;

                    int start_idx = 0;
                    int end_idx = 0;

                    if (lane_id == 0) {
                        start_idx = cell_start_idx[jcell];
                        end_idx = cell_start_idx[jcell + 1];
                    }

                    start_idx = tile.shfl(start_idx, 0);
                    end_idx = tile.shfl(end_idx, 0);

                    // TILE_SIZE毎のループ
                    for (int base = start_idx; base < end_idx; base += TILE_SIZE) {
                        const int j = base + lane_id;

                        bool is_neighbour = false;

                        if (j < end_idx && j != idx) {
                            const float pxj = pos.x[j];
                            const float pyj = pos.y[j];
                            const float pzj = pos.z[j];

                            float dx_pos = pxi - pxj;
                            float dy_pos = pyi - pyj;
                            float dz_pos = pzi - pzj;

                            cell.apply_pbc_device(&dx_pos, &dy_pos, &dz_pos);

                            const float dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos;

                            is_neighbour = dist_sq < cutoff_margin_sq;
                        }

                        const unsigned mask = tile.ballot(is_neighbour);

                        const int n_found = __popc(mask);

                        if (is_neighbour) {
                            // tile内で自分より前のneighbour数を数えるマスク
                            const unsigned lower = (1u << lane_id) - 1u;
                            const int offset = __popc(mask & lower);
                            const int dst = c + offset;

                            if (dst < max_neighbours) {
                                list[idx * max_neighbours + dst] = j;
                            }
                        }

                        c += n_found;
                    }
                }
            }
        }

        if (lane_id == 0) {
            count[idx] = c;
        }
    }
}

namespace md::neighbour {
    CellListNeighbourList::CellListNeighbourList(int n_atoms, int max_neighbours_, float cutoff_, float margin_, CellList& cl_) 
    : NeighbourList(n_atoms, max_neighbours_), cutoff(cutoff_), margin(margin_), cl(cl_) {}

    CellListNeighbourList::~CellListNeighbourList() {
        cudaFree(d_temp_storage);
    }

    void CellListNeighbourList::generate(State& state, SimState& simstate, Cell& cell) {
        auto N = state.n_atoms;
        auto cutoff_margin = cutoff + margin;
        auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

        // clの作成
        cl.generate(state, simstate, cell, flag);
        cl.sort(state, simstate, flag);

        // nlの作成
        int num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;
        
        generate_nl_kernel<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            N, 
            max_neighbours, 
            cl.get_M()[0], 
            cl.get_M()[1], 
            cl.get_M()[2], 
            thrust::raw_pointer_cast(list.data()), 
            thrust::raw_pointer_cast(count.data()), 
            cl.get_cell_id(), 
            cl.get_cell_start_idx(), 
            cutoff_margin_sq, 
            cell
        );

        update_nl_conf_kernel<<<num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            nl_conf, 
            N
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

    void CellListNeighbourList::check(State& state, SimState& simstate, Cell& cell) {
        auto N = state.n_atoms;
        auto cutoff_margin = cutoff + margin;
        auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

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

        cl.generate(state, simstate, cell, flag);
        cl.sort(state, simstate, flag);

        int generate_nl_num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;
        int update_nl_conf_num_blocks = (N + NUM_THREADS - 1) / NUM_THREADS;

        generate_nl_kernel<<<generate_nl_num_blocks, NUM_THREADS, 0, simstate.stream>>>(
            flag, 
            state.pos, 
            N, 
            max_neighbours, 
            cl.get_M()[0], 
            cl.get_M()[1], 
            cl.get_M()[2], 
            thrust::raw_pointer_cast(list.data()), 
            thrust::raw_pointer_cast(count.data()), 
            cl.get_cell_id(), 
            cl.get_cell_start_idx(), 
            cutoff_margin_sq, 
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