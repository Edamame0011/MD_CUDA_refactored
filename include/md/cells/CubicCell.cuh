#pragma once

#include <md/cells/Cell.cuh>

namespace md::cells {
    struct CubicCell : public Cell {
        CubicCell(std::array<std::array<float, 3>, 3>& _lattice);
        void apply_pbc(State& state) const override;
    };
}