#pragma once

#include <md/neighbour/NeighbourList.hpp>

namespace md {
    class CellList;
    class Cell;
    struct Top2;

    namespace neighbour {
        class CellListNeighbourListNoGraph : public NeighbourList {
            public:
                CellListNeighbourListNoGraph(int n_atoms, int max_neighbours, float cutoff, float margin, CellList* cl);
                ~CellListNeighbourListNoGraph();

                void generate(State& state, SimState& simstate, Cell& cell) override;
                void check(State& state, SimState& simstate, Cell& cell) override;

                CellListNeighbourListNoGraph(const NeighbourList&) = delete;
                CellListNeighbourListNoGraph& operator=(const NeighbourList&) = delete;

            private:
                CellList* cl;

                float cutoff, margin;

                bool flag = false;

                // cub用のバッファとそのサイズ
                void* d_temp_storage = nullptr;
                size_t temp_storage_bytes = 0;
            };
    }
}