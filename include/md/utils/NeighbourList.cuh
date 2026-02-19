#ifndef NEIGHBOUR_LIST_CUH
#define NEIGHBOUR_LIST_CUH

#include <md/core/State.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>
#include <vector_types.h>
#include <thrust/copy.h>

namespace {
    template <typename CellType>
    struct Generate {
        const float* d_x;
        const float* d_y;
        const float* d_z;
        float cutoff_plus_margin_sq;
        int num_atoms;
        CellType cell;
        
        Generate(const float* _d_x, const float* _d_y, const float* _d_z, int _num_atoms, float _cutoff_plus_margin_sq, CellType _cell) : d_x(_d_x), d_y(_d_y), d_z(_d_z), num_atoms(_num_atoms), cutoff_plus_margin_sq(_cutoff_plus_margin_sq), cell(_cell) {}
        __host__ __device__ bool operator() (const int idx) {
            int i = idx / num_atoms;
            int j = idx % num_atoms;

            if (j <= i) return false;

            // 距離の計算
            float x1 = d_x[i];
            float y1 = d_y[i];
            float z1 = d_z[i];

            float x2 = d_x[j];
            float y2 = d_y[j];
            float z2 = d_z[j];

            float dx = x1 - x2;
            float dy = y1 - y2;
            float dz = z1 - z2;

            // PBC補正
            cell.apply_pbc(dx, dy, dz);

            float dist_sq = dx * dx + dy * dy + dz * dz;

            return dist_sq < cutoff_plus_margin_sq;
        }
    };

    struct Top2 {
        float max1;
        float max2;

        __host__ __device__ Top2() : max1(-1.0f), max2(-1.0f) {}

        __host__ __device__ Top2(float v) : max1(v), max2(-1.0f) {}

        __host__ __device__ Top2(float m1, float m2) : max1(m1), max2(m2) {}
    };

    template <typename CellType>
    struct CalcDist {
        const float* d_x;
        const float* d_y;
        const float* d_z;

        const float* d_conf_x;
        const float* d_conf_y;
        const float* d_conf_z;

        CellType cell;

        CalcDist(const float* _d_x, const float* _d_y, const float* _d_z, const float* _d_conf_x, const float* _d_conf_y, const float* _d_conf_z, CellType _cell) : d_x(_d_x), d_y(_d_y), d_z(_d_z), d_conf_x(_d_conf_x), d_conf_y(_d_conf_y), d_conf_z(_d_conf_z), cell(_cell) {}

        __host__ __device__ Top2 operator () (const int idx) {
            float dx = d_x[idx] - d_conf_x[idx];
            float dy = d_y[idx] - d_conf_y[idx];
            float dz = d_z[idx] - d_conf_z[idx];

            // PBC補正
            cell.apply_pbc(dx, dy, dz);

            float dist_sq = dx * dx + dy * dy + dz * dz;

            return Top2(dist_sq);
        }
    };

    // 2つのTop2オブジェクトから新たな一つのTop2オブジェクトを作成
    struct MergeTop2 {
        __host__ __device__ Top2 operator () (const Top2& a, const Top2& b) {
            float vals[4] = {a.max1, a.max2, b.max1, b.max2};

            int max_idx = 0;
            for (int i = 1; i < 4; i ++) {
                if(vals[i] > vals[max_idx]) max_idx = i;
            }
            float new_max1 = vals[max_idx];
            vals[max_idx] = -1.0f;

            max_idx = 0;
            for (int i = 1; i < 4; i ++) {
                if(vals[i] > vals[max_idx]) max_idx = i;
            }
            float new_max2 = vals[max_idx];
            
            return Top2(new_max1, new_max2);
        }
    };
}

namespace md::utils { 
    class NeighbourList {
        public:
            NeighbourList(float _cutoff, float _margin) : cutoff(_cutoff), margin(_margin) {}

            template <typename CellType>
            void generate(State& state, CellType cell) {
                int num_atoms = state.n_atoms;

                // 十分なサイズのメモリを確保
                d_config.x.resize(num_atoms, 0);
                d_config.y.resize(num_atoms, 0);
                d_config.z.resize(num_atoms, 0);
                d_valid_indices.resize(num_atoms * num_atoms, 0);

                float* d_x = thrust::raw_pointer_cast(state.d_positions.x.data());
                float* d_y = thrust::raw_pointer_cast(state.d_positions.y.data());
                float* d_z = thrust::raw_pointer_cast(state.d_positions.z.data());

                float cutoff_plus_margin_sq = (cutoff + margin) * (cutoff + margin);

                auto end_ptr = thrust::copy_if(
                    thrust::make_counting_iterator(0), 
                    thrust::make_counting_iterator(num_atoms * num_atoms), 
                    d_valid_indices.begin(), 
                    Generate<CellType>(d_x, d_y, d_z, num_atoms, cutoff_plus_margin_sq, cell)
                );
            
                int NL_size = end_ptr - d_valid_indices.begin();
            
                d_valid_indices.resize(NL_size);
            
                // 座標をコピー
                d_config.x = state.d_positions.x;
                d_config.y = state.d_positions.y;
                d_config.z = state.d_positions.z;
            }

            template <typename CellType>
            void check(State& state, CellType cell) {
                int num_atoms = state.n_atoms;

                float* d_x = thrust::raw_pointer_cast(state.d_positions.x.data());
                float* d_y = thrust::raw_pointer_cast(state.d_positions.y.data());
                float* d_z = thrust::raw_pointer_cast(state.d_positions.z.data());

                // 最も移動距離が長い2粒子の移動距離を計算
                Top2 result = thrust::transform_reduce(
                    thrust::make_counting_iterator(0), 
                    thrust::make_counting_iterator(num_atoms), 
                    CalcDist<CellType>(
                        d_x, 
                        d_y, 
                        d_z, 
                        thrust::raw_pointer_cast(d_config.x.data()), 
                        thrust::raw_pointer_cast(d_config.y.data()), 
                        thrust::raw_pointer_cast(d_config.z.data()), 
                        cell
                    ), 
                    Top2(), 
                    MergeTop2()
                );
            
                if (result.max1 + result.max2 + 2.0f * std::sqrt(result.max1 * result.max2) > margin * margin) generate(state, cell); 
            }

            thrust::device_vector<int>& get_valid_indices() { return d_valid_indices; }

            float get_cutoff() { return cutoff; }
        
        private:
            float cutoff, margin;
            thrust::device_vector<int> d_valid_indices;
            struct device_float3 {
                thrust::device_vector<float> x, y, z;
            } d_config;
    };
}

#endif