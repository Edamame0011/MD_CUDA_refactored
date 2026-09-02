#pragma once

#include <md/neighbour/NeighbourList.hpp>

namespace md { 
    class Cell;

    namespace neighbour {
        class NativeNeighbourList : public NeighbourList {
            public:
                NativeNeighbourList(int n_atoms, int max_neighbours_, float cutoff_, float margin_);
                ~NativeNeighbourList();

                void generate(State& state, SimState& simstate, Cell& cell) override;
                void check(State& state, SimState& simstate, Cell& cell) override;

                NativeNeighbourList(const NeighbourList&) = delete;
                NativeNeighbourList& operator=(const NeighbourList&) = delete;
            private:
                float cutoff, margin;

                // cub用のバッファとそのサイズ
                void* d_temp_storage = nullptr;
                size_t temp_storage_bytes = 0;
            };
    }
}