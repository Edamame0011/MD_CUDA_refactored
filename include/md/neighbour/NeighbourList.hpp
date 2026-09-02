#pragma once

#include <thrust/device_vector.h>
#include <md/core/State.hpp>

namespace md { 
    class Cell;

    namespace neighbour {
        struct Top2;
    }
    
    class NeighbourList {
        public:
            virtual void generate(State& state, SimState& simstate, Cell& cell) = 0;
            virtual void check(State& state, SimState& simstate, Cell& cell) = 0;

            int* get_list() { return thrust::raw_pointer_cast(list.data()); }
            int* get_count() { return thrust::raw_pointer_cast(count.data()); }
            int get_max_neighbours() const { return this->max_neighbours; }

            virtual ~NeighbourList();

        protected:
            thrust::device_vector<int> list, count;
            int max_neighbours;
            DeviceVec3 nl_conf;
            md::neighbour::Top2* top2;
            bool* flag;

            NeighbourList(int n_atoms, int max_neighbours_);
        };
}