#include <md/utils/NeighbourList_CLL.cuh>
#include <md/utils/NeighbourList.cuh>
#include <cub/cub.cuh>

using Top2 = md::utils::Top2;
using CubicCell = md::cells::CubicCell;

namespace {
    __global__ void generate_nl_kernel(
        const bool* __restrict__ flag, 
        const dfloat3 sorted_pos, 
        const int num_atoms, 
        const int max_neighbours, 
        const int M, 
        unsigned int* __restrict__ list, 
        unsigned int* __restrict__ count, 
        const unsigned int* __restrict__ cell_id, 
        const unsigned int* __restrict__ cell_start_idx, 
        const float cutoff_margin_sq, 
        const CubicCell cell
    ) { 
        if (!*flag) return;

        int idx = threadIdx.x + blockDim.x * blockIdx.x;
        if (idx >= num_atoms) return;

        auto cid = cell_id[idx];

        auto pxi = sorted_pos.x[idx];
        auto pyi = sorted_pos.y[idx];
        auto pzi = sorted_pos.z[idx];

        int c = 0;

        // 自身の属するセルと隣接するセル内の隣接粒子を計算
        int cx = cid % M;
        int cy  = (cid / M) % M;
        int cz = cid / (M * M);

        for (int dz = -1; dz <= 1; dz ++) {
            for (int dy = -1; dy <= 1; dy ++) {
                for (int dx = -1; dx <= 1; dx ++) {
                    int jx = (cx + dx + M) % M;
                    int jy = (cy + dy + M) % M;
                    int jz = (cz + dz + M) % M;
                    int jcell = jx + jy * M + jz * M * M;

                    int start_idx = cell_start_idx[jcell];
                    int end_idx = cell_start_idx[jcell + 1];
                    for (int j = start_idx; j < end_idx; j ++) {
                        if (idx == j) continue;
                        auto pxj = sorted_pos.x[j];
                        auto pyj = sorted_pos.y[j];
                        auto pzj = sorted_pos.z[j];
                    
                        auto dx = pxi - pxj;
                        auto dy = pyi - pyj;
                        auto dz = pzi - pzj;
                    
                        cell.apply_pbc(dx, dy, dz);
                    
                        const auto dist_sq = dx * dx + dy * dy + dz * dz;
                    
                        if (dist_sq < cutoff_margin_sq) {
                            list[idx * max_neighbours + c] = j;
                            c ++; 
                        }
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

        CubicCell cell;

        CalcDist(
            dfloat3 _pos, 
            dfloat3 _nl_conf, 
            CubicCell _cell
        ) : pos(_pos), nl_conf(_nl_conf), cell(_cell) {}

        __device__ Top2 operator () (const int idx) const {
            auto dx = pos.x[idx] - nl_conf.x[idx];
            auto dy = pos.y[idx] - nl_conf.y[idx];
            auto dz = pos.z[idx] - nl_conf.z[idx];

            // PBC補正
            cell.apply_pbc(dx, dy, dz);

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

using namespace md::utils;

NeighbourList_CLL::NeighbourList_CLL(State& state, float _cutoff, float _margin, CellList& _cll) : cutoff(_cutoff), margin(_margin), cll(_cll) {
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
}

NeighbourList_CLL::~NeighbourList_CLL() {
    cudaFree(this->list);
    cudaFree(this->count);
    cudaFree(this->nl_conf.x);
    cudaFree(this->nl_conf.y);
    cudaFree(this->nl_conf.z);
    cudaFree(this->top2);
    cudaFree(this->flag);
    cudaFree(this->d_temp_storage);
}

void NeighbourList_CLL::generate(State& state, CubicCell cell) {
    auto N = state.n_atoms;
    auto view = state.get_view();
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // cllの作成
    cll.generate(state, flag);
    cll.sort(state, flag);

    // NLの作成
    int num_threads = 256;
    int num_blocks = (N + num_threads - 1) / num_threads;
    generate_nl_kernel<<<num_blocks, num_threads>>>(
        flag, 
        cll.get_sorted_pos(), 
        N, 
        max_neighbours, 
        cll.get_M(), 
        list, 
        count, 
        cll.get_cell_id(), 
        cll.get_cell_start_idx(), 
        cutoff_margin_sq, 
        cell
    );

    update_nl_conf_kernel<<<num_blocks, num_threads>>>(
        flag, 
        view.pos, 
        nl_conf, 
        N
    );

    cudaMemset(this->flag, 0, sizeof(bool));

    // バッファの確保
    CalcDist op(
        view.pos, 
        this->nl_conf, 
        cell
    );
    cub::CountingInputIterator<int> count_itr(0);
    cub::TransformInputIterator<Top2, CalcDist, cub::CountingInputIterator<int>> trans_itr(count_itr, op);

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

void NeighbourList_CLL::check(State& state, CubicCell cell) {
    auto view = state.get_view();
    auto N = state.n_atoms;
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // 移動距離の大きい順に2粒子の移動距離を表すTop2オブジェクトを計算
    CalcDist op(
        view.pos, 
        this->nl_conf, 
        cell
    );
    cub::CountingInputIterator<int> count_itr(0);
    cub::TransformInputIterator<Top2, CalcDist, cub::CountingInputIterator<int>> trans_itr(count_itr, op);

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
    cll.sort(state, flag);

    constexpr int num_threads = 256;

    int num_blocks = (N + num_threads - 1) / num_threads;

    generate_nl_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        flag, 
        cll.get_sorted_pos(), 
        N, 
        max_neighbours, 
        cll.get_M(), 
        list, 
        count, 
        cll.get_cell_id(), 
        cll.get_cell_start_idx(), 
        cutoff_margin_sq, 
        cell
    );

    update_nl_conf_kernel<<<num_blocks, num_threads, 0, state.stream>>>(
        flag, 
        view.pos, 
        nl_conf, 
        N
    );

    cudaMemsetAsync(this->flag, 0, sizeof(bool), state.stream);
}