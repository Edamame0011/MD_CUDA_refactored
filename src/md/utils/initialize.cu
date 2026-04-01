#include <md/utils/initialize.cuh>
#include <md/utils/compute.cuh>

#include <md/core/constant.h>

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <algorithm>

using namespace md::utils;

void initialize::init_velocities(State& state, float temperature, std::mt19937& mt) {
    // デバイスから質量を転送
    auto view = state.get_view();
    auto N = state.n_atoms;
    std::vector<float> h_mass(N);
    cudaMemcpy(h_mass.data(), view.mass, N * sizeof(float), cudaMemcpyDeviceToHost);

    // 平均0、分散1のガウス分布
    std::normal_distribution<float> dist_trans(0.0, 1);

    // ホスト側の配列
    std::vector<float> h_vel_x(N);
    std::vector<float> h_vel_y(N);
    std::vector<float> h_vel_z(N);

    for (int i = 0; i < N; i ++) {
        // 分散を調節
        h_vel_x[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / h_mass[i]);
        h_vel_y[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / h_mass[i]);
        h_vel_z[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / h_mass[i]);
    }

    // デバイスに速度を送信
    state.copy_vel(
        h_vel_x.data(), 
        h_vel_y.data(), 
        h_vel_z.data()
    );

    // 全体速度の除去
    md::utils::compute::remove_drift(state);
}

std::unique_ptr<md::State> initialize::generate_binary_lj(const int n_atoms, const float density, std::array<std::array<float, 3>, 3>& lattice, const float a_ratio, std::mt19937 &mt) {
    float Lbox = std::pow(n_atoms / density, 1.0f / 3.0f);
    lattice = {{{Lbox, 0, 0}, {0, Lbox, 0}, {0, 0, Lbox}}};
    std::vector<float> masses(n_atoms, 1.0f);
    std::vector<float> positions_x(n_atoms, 0.0f), positions_y(n_atoms, 0.0f), positions_z(n_atoms, 0.0f);
    std::vector<float> velocities_x(n_atoms, 0.0f), velocities_y(n_atoms, 0.0f), velocities_z(n_atoms, 0.0f);
    std::vector<float> forces_x(n_atoms, 0.0f), forces_y(n_atoms, 0.0f), forces_z(n_atoms, 0.0f);
    std::vector<int> box_x(n_atoms, 0), box_y(n_atoms, 0), box_z(n_atoms, 0);
    std::vector<int> atomic_numbers(n_atoms, 0);

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
    std::fill(atomic_numbers.begin(), atomic_numbers.begin() + num_a, 0);
    std::fill(atomic_numbers.begin() + num_a, atomic_numbers.end(), 1);
    std::shuffle(atomic_numbers.begin(), atomic_numbers.end(), mt);

    auto state = std::make_unique<md::State>(n_atoms);

    // GPUに転送
    state->copy(
        positions_x.data(), positions_y.data(), positions_z.data(), 
        velocities_x.data(), velocities_y.data(), velocities_z.data(), 
        forces_x.data(), forces_y.data(), forces_z.data(), 
        masses.data(), atomic_numbers.data()
    );

    return state;
}