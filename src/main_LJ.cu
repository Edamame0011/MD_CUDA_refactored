#include <md/core/State.cuh>
#include <md/core/Simulator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/interactions/LJPotential.cuh>
#include <md/observers/LinearOutput.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/initialize.cuh>

int main() {
    // 定数の定義
    constexpr float dt = 0.5;
    constexpr float cutoff = 5.0;
    constexpr float margin = 1.0;
    constexpr float temperature = 300;
    constexpr int output_interval = 100;

    const std::string data_path = "./data/sample_NS2.xyz";
    const std::string model_path = "./models/deployed_model_Na2O-SiO2.pt";
    const std::string json_path = "./configs/example.json";

    std::mt19937 mt(123456789);
    md::State state;

    // 系の初期化
    md::utils::initialize::read_state_from_xyz(state, data_path);  // 座標の初期化
    md::utils::initialize::init_velocities(state, temperature, mt);    // 速度の初期化
    state.dt = dt;

    // シミュレーションセルの初期化
    std::array<std::array<float, 3>, 3> lattice;
    lattice = md::utils::initialize::find_lattice_from_xyz(data_path);
    md::cells::CubicCell cell(lattice[0][0]);

    // 隣接リストの作成
    md::utils::NeighbourList NL(cutoff, margin);
    NL.generate(state, cell);

    // 相互作用ポテンシャルの初期化
    md::interactions::LJPotential interaction = md::utils::initialize::init_LJPotential_from_json(json_path, state, cell, &NL);

    // 熱浴の初期化
    md::thermostats::NoThermostat no_thermostat;

    // integratorの初期化
    md::integrators::ConstantVolume integrator(&no_thermostat);

    // observerの初期化
    md::observers::LinearOutput observer(output_interval);

    // simulatorの初期化
    md::Simulator simulator(state, &interaction, &integrator, &observer, cell);

    // シミュレーションの実行
    simulator.run(5e+3f);
}