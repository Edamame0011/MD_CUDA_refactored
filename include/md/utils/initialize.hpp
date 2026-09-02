#pragma once

#include <memory>

namespace std {
    class mt19937;
}

namespace md {
    class State;
    class Cell;

    namespace utils {
        std::unique_ptr<md::State> generate_binary_lj(const int n_atoms, const float density, std::unique_ptr<Cell>& cell, const float a_ratio, std::mt19937 &mt);
    }
}