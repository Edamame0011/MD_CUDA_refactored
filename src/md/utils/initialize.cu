#include <md/utils/initialize.hpp>
#include <md/core/State.hpp>

#include <vector>
#include <algorithm>

namespace md::utils {
    std::unique_ptr<State> generate_binary_lj(const int n_atoms, const float density, std::unique_ptr<Cell>& cell, const float a_ratio, std::mt19937 &mt) {
        float Lbox = std::pow(n_atoms / density, 1.0f / 3.0f);
        std::array<float, 3> lattice = {Lbox, Lbox, Lbox};
        std::vector<float> masses(n_atoms, 1.0f);
        std::vector<float> positions_x(n_atoms, 0.0f), positions_y(n_atoms, 0.0f), positions_z(n_atoms, 0.0f);
        std::vector<float> velocities_x(n_atoms, 0.0f), velocities_y(n_atoms, 0.0f), velocities_z(n_atoms, 0.0f);
        std::vector<float> forces_x(n_atoms, 0.0f), forces_y(n_atoms, 0.0f), forces_z(n_atoms, 0.0f);
        std::vector<int> box_x(n_atoms, 0), box_y(n_atoms, 0), box_z(n_atoms, 0);
        std::vector<int> species(n_atoms, 0);

        cell = std::make_unique<Cell>(lattice);

        // 位置の初期化
        const auto ln = static_cast<int>(std::ceil(std::pow(n_atoms, 1.0f / 3.0f)));
        const auto haba = Lbox / ln;

        for (int i = 0; i < n_atoms; i ++) {
            const int iz = static_cast<int>(std::floor(i / (ln * ln)));
            const int iy = static_cast<int>(std::floor((i - iz * ln * ln) / ln));
            const int ix = i - iz * ln * ln - iy * ln;

            positions_x[i] = haba * 0.5f + haba * ix;
            positions_y[i] = haba * 0.5f + haba * iy;
            positions_z[i] = haba * 0.5f + haba * iz;
        }

        // 種類の初期化
        std::size_t num_a = static_cast<std::size_t>(n_atoms * a_ratio);
        std::fill(species.begin(), species.begin() + num_a, 0);
        std::fill(species.begin() + num_a, species.end(), 1);
        std::shuffle(species.begin(), species.end(), mt);

        auto state = std::make_unique<md::State>(n_atoms);

        // GPUに転送
        state->init(
            positions_x.data(), positions_y.data(), positions_z.data(), 
            velocities_x.data(), velocities_y.data(), velocities_z.data(), 
            forces_x.data(), forces_y.data(), forces_z.data(), 
            masses.data(), species.data()
        );

        return state;
    }
}