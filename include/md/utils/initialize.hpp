#pragma once

#include <memory>
#include <random>
#include <external/nlohmann/json.hpp>

namespace md {
    class State;
    class Interaction;
    class NeighbourList;
    class Cell;

    namespace utils {
        std::unique_ptr<md::State> generate_binary_lj(const int n_atoms, const float density, std::unique_ptr<Cell>& cell, const float a_ratio, std::mt19937 &mt);
        std::unique_ptr<md::Interaction> build_lj_potential(const nlohmann::json& json, State &state, Cell* cell, NeighbourList* NL);
        void init_velocities(State& state, float temperature, std::mt19937& mt);
    }
}