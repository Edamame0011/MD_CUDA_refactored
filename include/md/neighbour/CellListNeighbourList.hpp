#pragma once

#include <md/neighbour/NeighbourList.hpp>

namespace md {
    class CellList;
    class Cell;
    struct Top2;

    namespace neighbour {
        class CellListNeighbourList : public NeighbourList {
            public:
                CellListNeighbourList(int n_atoms, int max_neighbours, float cutoff, float margin, CellList& cl);
                ~CellListNeighbourList();

                void generate(State& state, SimState& simstate, Cell& cell) override;
                void check(State& state, SimState& simstate, Cell& cell) override;

                CellListNeighbourList(const NeighbourList&) = delete;
                CellListNeighbourList& operator=(const NeighbourList&) = delete;

            private:
                CellList& cl;

                float cutoff, margin;

                // cub用のバッファとそのサイズ
                void* d_temp_storage = nullptr;
                size_t temp_storage_bytes = 0;
            };
    }
}