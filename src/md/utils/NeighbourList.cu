#include <md/utils/NeighbourList.cuh>
#include <md/cells/CubicCell.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <cub/cub.cuh>

using Top2 = md::utils::Top2;

namespace {
    template <typename CellType>
    __global__ void generate_nl(
        bool* flag, 
        dfloat3 pos, 
        dfloat3 nl_conf, 
        int num_atoms, 
        int max_neighbours, 
        int* list, 
        int* count, 
        float cutoff_margin_sq, 
        CellType cell
    ) {
        if (!*flag) return;

        int idx = blockDim.x * blockIdx.x + threadIdx.x;

        if (idx < num_atoms) {
            auto pxi = pos.x[idx];
            auto pyi = pos.y[idx];
            auto pzi = pos.z[idx];

            int c = 0;
            for (int j = 0; j < num_atoms; j ++) {
                if (idx == j) continue;

                auto pxj = pos.x[j];
                auto pyj = pos.y[j];
                auto pzj = pos.z[j];

                // 距離の計算
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
            count[idx] = c;
            nl_conf.x[idx] = pxi;
            nl_conf.y[idx] = pyi;
            nl_conf.z[idx] = pzi;
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

    template <typename CellType>
    struct CalcDist {
        dfloat3 pos;
        dfloat3 nl_conf;

        CellType cell;

        CalcDist(
            dfloat3 _pos, 
            dfloat3 _nl_conf, 
            CellType _cell
        ) : pos(_pos), nl_conf(_nl_conf), cell(_cell) {}

        __host__ __device__ Top2 operator () (const int idx) const {
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

template <typename CellType>
NeighbourList<CellType>::NeighbourList(State& state, float _cutoff, float _margin) : cutoff(_cutoff), margin(_margin) {
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

template <typename CellType>
NeighbourList<CellType>::~NeighbourList() {
    cudaFree(this->list);
    cudaFree(this->count);
    cudaFree(this->nl_conf.x);
    cudaFree(this->nl_conf.y);
    cudaFree(this->nl_conf.z);
    cudaFree(this->top2);
    cudaFree(this->flag);
    cudaFree(this->d_temp_storage);
}

template <typename CellType>
void NeighbourList<CellType>::generate(State& state, CellType cell) {
    auto N = state.n_atoms;
    auto view = state.get_view();
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // NLの作成
    int num_threads = 128;
    int num_blocks = (N + num_threads - 1) / num_threads;
    generate_nl<CellType><<<num_blocks, num_threads>>>(
        this->flag, 
        view.pos, 
        this->nl_conf, 
        N, 
        this->max_neighbours, 
        this->list, 
        this->count, 
        cutoff_margin_sq, 
        cell
    );

    cudaMemset(this->flag, 0, sizeof(bool));

    // バッファの確保
    CalcDist<CellType> op(
        view.pos, 
        this->nl_conf, 
        cell
    );
    cub::CountingInputIterator<int> count_itr(0);
    cub::TransformInputIterator<Top2, CalcDist<CellType>, cub::CountingInputIterator<int>> trans_itr(count_itr, op);

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

template <typename CellType>
void NeighbourList<CellType>::check(State& state, CellType cell) {
    auto view = state.get_view();
    auto N = state.n_atoms;
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_sq = cutoff_margin * cutoff_margin;

    // 移動距離の大きい順に2粒子の移動距離を表すTop2オブジェクトを計算
    CalcDist<CellType> op(
        view.pos, 
        this->nl_conf, 
        cell
    );
    cub::CountingInputIterator<int> count_itr(0);
    cub::TransformInputIterator<Top2, CalcDist<CellType>, cub::CountingInputIterator<int>> trans_itr(count_itr, op);

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

    int num_threads = 128;
    int num_blocks = (N + num_threads - 1) / num_threads;
    generate_nl<CellType><<<num_blocks, num_threads, 0, state.stream>>>(
        this->flag, 
        view.pos, 
        this->nl_conf, 
        N, 
        this->max_neighbours, 
        this->list, 
        this->count, 
        cutoff_margin_sq, 
        cell
    );

    cudaMemsetAsync(this->flag, 0, sizeof(bool), state.stream);
}

template class NeighbourList<md::cells::CubicCell>;