#ifndef INITIALIZE_FROM_JSON_CUH
#define INITIALIZE_FROM_JSON_CUH

#include <md/core/State.cuh>
#include <string>
#include <random>
#include <array>
#include <random>
#include <external/nlohmann/json.hpp>

using json = nlohmann::json;

namespace md::utils::initialize {
    std::array<std::array<float, 3>, 3> init_state(md::State& state, const json& a_setting, std::mt19937& mt);
}

#endif