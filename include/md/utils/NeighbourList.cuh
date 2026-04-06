#ifndef NEIGHBOUR_LIST_CUH
#define NEIGHBOUR_LIST_CUH

#include <md/core/State.cuh>

namespace md::utils { 
    struct Top2 {
        float max1, max2;

        __host__ __device__ Top2() : max1(0.0f), max2(0.0f) {}
        __host__ __device__ Top2(float m1) : max1(m1), max2(0.0f) {}
        __host__ __device__ Top2(float m1, float m2) : max1(m1), max2(m2) {}
    };

    template <typename CellType>
    class NeighbourList {
        public:
            NeighbourList(State& state, float _cutoff, float _margin);
            ~NeighbourList();

            void generate(State& state, CellType cell);
            void check(State& state, CellType cell);

            int* get_list() { return this->list; }
            int* get_count() { return this->count; }
            int get_max_neighbours() { return this->max_neighbours; }
        
        private:
            float cutoff, margin;
            dfloat3 nl_conf;
            int* list;
            int* count;
            size_t max_neighbours;

            Top2* top2;
            bool* flag;

            // cub用のバッファとそのサイズ
            void* d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
        };
}

#endif