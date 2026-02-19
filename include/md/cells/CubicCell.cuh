#ifndef CUBIC_CELL_CUH
#define CUBIC_CELL_CUH

#include <md/core/State.cuh>

namespace md::cells {
    struct CubicCell {
        float Lbox;

        CubicCell(float _Lbox) : Lbox(_Lbox) {}

        void apply_pbc(State& state) const;
        __device__ __host__ void apply_pbc(float& x, float& y, float& z) const {
            float Linv = 1.0f / Lbox;
                
            x -= Lbox * floorf(x * Linv + 0.5f);
            y -= Lbox * floorf(y * Linv + 0.5f);
            z -= Lbox * floorf(z * Linv + 0.5f);
        }
    };
}

#endif