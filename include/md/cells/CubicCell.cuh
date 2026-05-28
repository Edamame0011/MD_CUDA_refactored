#ifndef CUBIC_CELL_CUH
#define CUBIC_CELL_CUH

#include <md/core/State.cuh>
#include <array>

namespace md::cells {
    struct CubicCell {
        float Lbox;

        CubicCell(std::array<std::array<float, 3>, 3> _lattice) : Lbox(_lattice[0][0]) {}

        void apply_pbc(State& state) const;
        __forceinline__ __device__ void apply_pbc(float& x, float& y, float& z) const {
            float Linv = 1.0f / Lbox;
                
            x -= Lbox * floorf(x * Linv + 0.5f);
            y -= Lbox * floorf(y * Linv + 0.5f);
            z -= Lbox * floorf(z * Linv + 0.5f);
        }

        void init (std::array<std::array<float, 3>, 3> lattice) {
            this->Lbox = lattice[0][0];
        }

        std::array<std::array<float, 3>, 3> get_lattice() const {
            std::array<std::array<float, 3>, 3> lattice = {{
                {Lbox, 0, 0}, 
                {0, Lbox, 0}, 
                {0, 0, Lbox}
            }};

            return lattice;
        }
    };
}

#endif