#pragma once

#include <md/neighbour/NeighbourList.hpp>
#include <vector>

namespace md { 
    class Cell;

    namespace neighbour {
        class NativeNeighbourList : public NeighbourList {
            public:
                NativeNeighbourList(int n_atoms, int n_species, int max_neighbours, std::vector<float> cutoff, float margin);
                ~NativeNeighbourList();

                void generate(State* state, SimState& simstate, Cell& cell) override;
                void check(State* state, SimState& simstate, Cell& cell) override;

                NativeNeighbourList(const NeighbourList&) = delete;
                NativeNeighbourList& operator=(const NeighbourList&) = delete;
            private:
                bool* flag;

                // cub用のバッファとそのサイズ
                void* d_temp_storage = nullptr;
                size_t temp_storage_bytes = 0;
            };
    }
}