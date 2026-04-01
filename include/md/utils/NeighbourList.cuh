#ifndef NEIGHBOUR_LIST_CUH
#define NEIGHBOUR_LIST_CUH

#include <md/core/State.cuh>

namespace md::utils { 
    template <typename CellType>
    class NeighbourList {
        public:
            NeighbourList(State& state, float _cutoff, float _margin);
            ~NeighbourList();

            void generate(State& state, CellType cell);
            void check(State& state, CellType cell);

            int* get_list() { return this->list; }
            int* get_count() { return this->count; }
        
        private:
            float cutoff, margin;
            dfloat3 nl_conf;
            int* list;
            int* count;
        };
}

#endif