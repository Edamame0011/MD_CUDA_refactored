#include <md/utils/initialize.hpp>
#include <md/core/State.hpp>
#include <md/core/constant.h>
#include <md/core/Cell.cuh>
#include <md/interactions/LJPotential.hpp>

#include <vector>
#include <algorithm>
#include <map>
#include <numeric>

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

        cell = std::make_unique<Cell>(n_atoms, lattice);

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

    void init_velocities(State& state, float temperature, std::mt19937& mt) {
        // デバイスから質量を転送
        auto N = state.n_atoms;
        std::vector<float> h_mass(N);
        cudaMemcpy(h_mass.data(), state.mass, N * sizeof(float), cudaMemcpyDeviceToHost);

        // 平均0、分散1のガウス分布
        std::normal_distribution<float> dist_trans(0.0, 1);

        // ホスト側の配列
        std::vector<float> h_vel_x(N);
        std::vector<float> h_vel_y(N);
        std::vector<float> h_vel_z(N);

        for (int i = 0; i < N; i ++) {
            const float sigma = std::sqrt((boltzmann_constant * temperature * conversion_factor) / h_mass[i]);
            // 分散を調節
            h_vel_x[i] = dist_trans(mt) * sigma;
            h_vel_y[i] = dist_trans(mt) * sigma;
            h_vel_z[i] = dist_trans(mt) * sigma;
        }

        // 全体速度の除去
        float total_mass = 0.0f;
        float momentum_x = 0.0f;
        float momentum_y = 0.0f;
        float momentum_z = 0.0f;

        for (int i = 0; i < N; i ++) {
            const float m = h_mass[i];

            total_mass += m;
            momentum_x += m * h_vel_x[i];
            momentum_y += m * h_vel_y[i];
            momentum_z += m * h_vel_z[i];
        }

        const float vel1 = momentum_x / total_mass;
        const float vel2 = momentum_y / total_mass;
        const float vel3 = momentum_z / total_mass;
        
        std::transform(h_vel_x.begin(), h_vel_x.end(), h_vel_x.begin(), [vel1] (float v) { return v - vel1; });
        std::transform(h_vel_y.begin(), h_vel_y.end(), h_vel_y.begin(), [vel2] (float v) { return v - vel2; });
        std::transform(h_vel_z.begin(), h_vel_z.end(), h_vel_z.begin(), [vel3] (float v) { return v - vel3; });

        // デバイスに速度を送信
        state.copy_vel(
            h_vel_x.data(), 
            h_vel_y.data(), 
            h_vel_z.data()
        );        
    }
}