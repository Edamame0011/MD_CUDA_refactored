#pragma once

#include <md/neighbour/CellList.hpp>

namespace md {
    class State;
    class SimState;
    class Cell;

    class CellListNoGraph : public CellList {
        public:
            using CellList::CellList;

            void generate(State* state, SimState& simstate, Cell& cell, bool* flag) override;
            void sort(State* state, SimState& simstate, bool* flag) override;
    };
}