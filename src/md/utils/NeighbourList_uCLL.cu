#include <md/utils/NeighbourList_uCLL.cuh>

#include <md/core/State.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/cells/Cell.cuh>
#include <md/utils/UnsortedCellList.cuh>

#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>

using Top2 = md::Top2;
using Cell = md::Cell;

namespace {
    __global__ void generate_nl_kernel(
        const bool* __restrict__ flag, 
        const dfloat3 pos, 
        const int num_atoms, 
        const int max_neighbours, 
        const int Mx, 
        const int My, 
        const int Mz, 
        const float cell_size_inv_x, 
        const float cell_size_inv_y, 
        const float cell_size_inv_z, 
        int* __restrict__ list, 
        int* __restrict__ count, 
        const int* __restrict__ head, 
        const int* __restrict__ next, 
        const float cutoff_margin_sq, 
        Cell cell
    ) { 
        if (!*flag) return;

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto pxi = pos.x[idx];
        auto pyi = pos.y[idx];
        auto pzi = pos.z[idx];

        // 自身の属するセルと隣接するセル内の隣接粒子を計算
        float px = pxi;
        float py = pyi;
        float pz = pzi;

        cell.apply_pbc_wrap_device(&px, &py, &pz);

        int cx = max(0, min(Mx - 1, (int)(px * cell_size_inv_x)));
        int cy = max(0, min(My - 1, (int)(py * cell_size_inv_y)));
        int cz = max(0, min(Mz - 1, (int)(pz * cell_size_inv_z)));

        int c = 0;

        for (int dz = -1; dz <= 1; dz ++) {
            for (int dy = -1; dy <= 1; dy ++) {
                for (int dx = -1; dx <= 1; dx ++) {
                    int jx = (cx + dx + Mx) % Mx;
                    int jy = (cy + dy + My) % My;
                    int jz = (cz + dz + Mz) % Mz;
                    int jcell = jx + jy * Mx + jz * Mx * My;

                    int j = head[jcell];
                    while (j != -1) {
                        if (idx != j) {
                            auto pxj = pos.x[j];
                            auto pyj = pos.y[j];
                            auto pzj = pos.z[j];
                        
                            auto dx_pos = pxi - pxj;
                            auto dy_pos = pyi - pyj;
                            auto dz_pos = pzi - pzj;
                        
                            cell.apply_pbc_device(&dx_pos, &dy_pos, &dz_pos);

                            const auto dist_sq = dx_pos * dx_pos + dy_pos * dy_pos + dz_pos * dz_pos;
                        
                            if (dist_sq < cutoff_margin_sq) {
                                list[idx * max_neighbours + c] = j;
                                c ++; 
                            }
                        }
                        j = next[j];
                    }
                }
            }
        }

        count[idx] = c;
    }

    __global__ void update_nl_conf_kernel(
        const bool* flag, 
        const dfloat3 pos, 
        dfloat3 nl_conf, 
        int num_atoms
    ) {
        if (!*flag) return;
        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx < num_atoms) {
            nl_conf.x[idx] = pos.x[idx];
            nl_conf.y[idx] = pos.y[idx];
            nl_conf.z[idx] = pos.z[idx];
        }
    }

    __global__ void check_top2(
        Top2* top2, 
        bool* flag, 
        float margin_sq
    ) {
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            Top2 t = *top2;
            if (t.max1 + t.max2 + 2 * sqrtf(t.max1 * t.max2) > margin_sq) *flag = true;
        }
    }

    struct CalcDist {
        dfloat3 pos;
        dfloat3 nl_conf;

        Cell cell;

        CalcDist(
            dfloat3 _pos, 
            dfloat3 _nl_conf, 
            Cell _cell
        ) : pos(_pos), nl_conf(_nl_conf), cell(_cell) {}

        __device__ Top2 operator () (const int idx) const {
            auto dx = pos.x[idx] - nl_conf.x[idx];
            auto dy = pos.y[idx] - nl_conf.y[idx];
            auto dz = pos.z[idx] - nl_conf.z[idx];

            // PBC補正
            cell.apply_pbc_device(&dx, &dy, &dz);

            float dist_sq = dx * dx + dy * dy + dz * dz;

            return Top2(dist_sq);
        }
    };

    // 2つのTop2オブジェクトから新たな一つのTop2オブジェクトを作成
    struct MergeTop2 {
        __host__ __device__ Top2 operator () (const Top2& a, const Top2& b) const {
            float max1 = fmaxf(a.max1, b.max1);
            float max2 = fmaxf(fminf(a.max1, b.max1), fmaxf(a.max2, b.max2));
            return Top2(max1, max2);        
        }
    };
}

using namespace md;

NeighbourList_uCLL::NeighbourList_uCLL(State& state, float _cutoff, float _margin, UnsortedCellList& _cll) : cutoff(_cutoff), margin(_margin), cll(_cll) {
    auto N = state.n_atoms;
    this->max_neighbours = 1000;
    cudaMalloc(&this->list, (size_t)N * max_neighbours * sizeof(int));
    cudaMalloc(&this->count, N * sizeof(int));
    cudaMalloc(&this->nl_conf.x, N * sizeof(float));
    cudaMalloc(&this->nl_conf.y, N * sizeof(float));
    cudaMalloc(&this->nl_conf.z, N * sizeof(float));
    cudaMalloc(&this->top2, sizeof(Top2));
    cudaMalloc(&this->flag, sizeof(bool));
    cudaMemset(this->flag, 1, sizeof(bool));

    int minGridSize;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &generate_nl_num_threads, generate_nl_kernel, 0, 0);
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &update_nl_conf_num_threads, update_nl_conf_kernel, 0, 0);
}

NeighbourList_uCLL::~NeighbourList_uCLL() {
    cudaFree(this->list);
    cudaFree(this->count);
    cudaFree(this->nl_conf.x);
    cudaFree(this->nl_conf.y);
    cudaFree(this->nl_conf.z);
    cudaFree(this->top2);
    cudaFree(this->flag);
    cudaFree(this->d_temp_storage);
}

void NeighbourList_uCLL::generate(State& state, Cell* cell) {
    auto N = state.n_atoms;
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // cllの作成
    cll.generate(state, flag);

    // NLの作成
    int generate_nl_num_blocks = (N + generate_nl_num_threads - 1) / generate_nl_num_threads;
    int update_nl_conf_num_blocks = (N + update_nl_conf_num_threads - 1) / update_nl_conf_num_threads;

    generate_nl_kernel<<<generate_nl_num_blocks, generate_nl_num_threads>>>(
        flag, 
        state.pos, 
        N, 
        max_neighbours, 
        cll.get_M()[0], 
        cll.get_M()[1], 
        cll.get_M()[2], 
        1.0f / cll.get_cell_size()[0], 
        1.0f / cll.get_cell_size()[1], 
        1.0f / cll.get_cell_size()[2],  
        list, 
        count, 
        cll.get_head(), 
        cll.get_next(), 
        cutoff_margin_sq, 
        *cell
    );

    update_nl_conf_kernel<<<update_nl_conf_num_blocks, update_nl_conf_num_threads>>>(
        flag, 
        state.pos, 
        nl_conf, 
        N
    );

    cudaMemset(this->flag, 0, sizeof(bool));

    // バッファの確保
    CalcDist op(
        state.pos, 
        this->nl_conf, 
        *cell
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

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

void NeighbourList_uCLL::check(State& state, Cell* cell) {
    auto N = state.n_atoms;
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // 移動距離の大きい順に2粒子の移動距離を表すTop2オブジェクトを計算
    CalcDist op(
        state.pos, 
        this->nl_conf, 
        *cell
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
        state.stream
    );

    // Top2オブジェクトの移動距離が閾値を超えているか判定し、超えていたらGenerate()を呼ぶ
    check_top2<<<1, 1, 0, state.stream>>>(
        this->top2, 
        this->flag, 
        margin * margin
    );

    cll.generate(state, flag);

    int generate_nl_num_blocks = (N + generate_nl_num_threads - 1) / generate_nl_num_threads;
    int update_nl_conf_num_blocks = (N + update_nl_conf_num_threads - 1) / update_nl_conf_num_threads;

    generate_nl_kernel<<<generate_nl_num_blocks, generate_nl_num_threads, 0, state.stream>>>(
        flag, 
        state.pos, 
        N, 
        max_neighbours, 
        cll.get_M()[0], 
        cll.get_M()[1], 
        cll.get_M()[2], 
        1.0f / cll.get_cell_size()[0], 
        1.0f / cll.get_cell_size()[1], 
        1.0f / cll.get_cell_size()[2],  
        list, 
        count, 
        cll.get_head(), 
        cll.get_next(), 
        cutoff_margin_sq, 
        *cell
    );

    update_nl_conf_kernel<<<update_nl_conf_num_blocks, update_nl_conf_num_threads, 0, state.stream>>>(
        flag, 
        state.pos, 
        nl_conf, 
        N
    );

    cudaMemsetAsync(this->flag, 0, sizeof(bool), state.stream);
}