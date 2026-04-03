#include <md/utils/NeighbourList.cuh>
#include <md/cells/CubicCell.cuh>
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>

namespace {
    template <typename CellType>
    struct Generate {
        dfloat3 pos, nl_conf;
        int num_atoms;
        int max_neighbours;
        int* list; 
        int* count;
        float cutoff_margin_sq;
        CellType cell;

        Generate(
            dfloat3 _pos, 
            int _num_atoms, 
            int _max_neighbours, 
            dfloat3 _nl_conf, 
            int* _list, 
            int* _count, 
            float _cutoff_margin_sq, 
            CellType _cell
        ) : pos(_pos), num_atoms(_num_atoms), max_neighbours(_max_neighbours), nl_conf(_nl_conf), list(_list), count(_count), cutoff_margin_sq(_cutoff_margin_sq), cell(_cell) {}

        __device__ void operator() (int idx) {
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
    };

    struct Top2 {
        float max1, max2;

        __host__ __device__ Top2() : max1(-1.0f), max2(-1.0f) {}

        __host__ __device__ Top2(float v) : max1(v), max2(-1.0f) {}

        __host__ __device__ Top2(float m1, float m2) : max1(m1), max2(m2) {}
    };

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

        __host__ __device__ Top2 operator () (const int idx) {
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
        __host__ __device__ Top2 operator () (const Top2& a, const Top2& b) {
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
    this->max_neighbours = 10000;
    cudaMalloc(&this->list, (size_t)N * max_neighbours * sizeof(int));
    cudaMalloc(&this->count, N * sizeof(int));
    cudaMalloc(&this->nl_conf.x, N * sizeof(float));
    cudaMalloc(&this->nl_conf.y, N * sizeof(float));
    cudaMalloc(&this->nl_conf.z, N * sizeof(float));
}

template <typename CellType>
NeighbourList<CellType>::~NeighbourList() {
    cudaFree(this->list);
    cudaFree(this->count);
    cudaFree(this->nl_conf.x);
    cudaFree(this->nl_conf.y);
    cudaFree(this->nl_conf.z);
}

template <typename CellType>
void NeighbourList<CellType>::generate(State& state, CellType cell) {
    auto N = state.n_atoms;
    auto view = state.get_view();
    auto cutoff_margin = cutoff + margin;
    auto cutoff_margin_2 = cutoff_margin * cutoff_margin;

    thrust::for_each(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        Generate<CellType>(
            view.pos, 
            N, 
            this->max_neighbours, 
            this->nl_conf, 
            this->list, 
            this->count, 
            cutoff_margin_2, 
            cell
        )
    );
}

template <typename CellType>
void NeighbourList<CellType>::check(State& state, CellType cell) {
    auto view = state.get_view();
    auto N = state.n_atoms;

    Top2 result = thrust::transform_reduce(
        thrust::device, 
        thrust::make_counting_iterator(0), 
        thrust::make_counting_iterator(N), 
        CalcDist<CellType>(
            view.pos, 
            this->nl_conf, 
            cell
        ), 
        Top2(), 
        MergeTop2()
    );

    if (result.max1 + result.max2 + 2.0f * std::sqrt(result.max1 * result.max2) > margin * margin) generate(state, cell); 
}

template class NeighbourList<md::cells::CubicCell>;