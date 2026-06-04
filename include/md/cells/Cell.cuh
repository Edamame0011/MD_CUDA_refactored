#pragma once

#include <array>

namespace md {
    class State;
}

namespace md{
    class Cell {
        public:
            virtual ~Cell() {
                cudaFree(d_lattice);
            }

            virtual void apply_pbc(State& state) const = 0;

            void (*apply_pbc_ptr) (float*, float*, float*, float*);
            std::array<std::array<float, 3>, 3> lattice;
            float* d_lattice;

        protected:
            Cell(const std::array<std::array<float, 3>, 3>& _lattice) : lattice(_lattice) {
                cudaMalloc(&d_lattice, 3 * 3 * sizeof(float));
            }

            Cell(const Cell&) = delete;
            Cell& operator=(const Cell&) = delete;
    };
}