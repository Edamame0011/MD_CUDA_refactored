#include <md/utils/SimulationRunner.hpp>

#include <md/core/State.hpp>
#include <md/integrators/Integrator.hpp>
#include <md/interactions/Interaction.hpp>
#include <md/observers/Observer.hpp>
#include <md/core/Cell.cuh>
// #include <md/temperature_schedulers/TemperatureScheduler.hpp>
#include <md/thermostats/Thermostat.cuh>
#include <md/neighbour/NeighbourList.hpp>
// #include <md/convergence_checkers/ConvChecker.hpp>
// #include <md/energy_minimizers/EnergyMinimizer.hpp>

#include <md/core/constant.h>
#include <md/core/Simulator.hpp>
// #include <md/integrators/ConstantVolume.hpp>
#include <md/interactions/LJPotential.hpp>
// #include <md/interactions/NNP.hpp>
// #include <md/integrators/LangevinIntegrator.hpp>
// #include <md/observers/LinearOutput.hpp>
// #include <md/thermostats/NoThermostat.hpp>
// #include <md/thermostats/NHC1.hpp>
// #include <md/thermostats/BussiThermostat.hpp>
// #include <md/utils/NeighbourList.hpp>
// #include <md/utils/SortedCellList.hpp>
// #include <md/utils/UnsortedCellList.hpp>
// #include <md/observers/LogOutput.hpp>
// #include <md/temperature_schedulers/TemperatureScheduler.hpp>
// #include <md/temperature_schedulers/ConstantScheduler.hpp>
// #include <md/temperature_schedulers/LinearScheduler.hpp>
// #include <md/observers/LinearExportTrajectory.hpp>
// #include <md/observers/LogExportTrajectory.hpp>
// #include <md/observers/TargetTemperatureExporter.hpp>
// #include <md/observers/LogplusStrideExportTrajectory.hpp>
// #include <md/convergence_checkers/MaxNorm.hpp>
// #include <md/energy_minimizers/FireMinimizer.hpp>
// #include <md/thermostats/KinEnergyCalculator.hpp>
// #include <md/observers/TrajectoryExporter.hpp>

#include <cmath>
#include <fstream>
#include <iostream>

using namespace md::utils;
using namespace md;
namespace fs = std::filesystem;

using string = std::string;
using json = nlohmann::json;

SimulationRunner::SimulationRunner(const string& setting_path) {
    // jsonのロード
    std::ifstream f(setting_path);
    if (!f.is_open()) throw std::runtime_error("jsonファイルを開けません。" );
    this->j = json::parse(f);

    const json& m_setting = j.at("meta");
    const json& c_setting = j.at("common_settings");

    // 親ディレクトリの作成
    parent_dir = m_setting.at("name").get<string>();
    fs::create_directories(parent_dir);

    // 親ディレクトリにjsonをコピーしておく
    fs::copy_file(
        setting_path, 
        parent_dir / "setting.json", 
        fs::copy_options::overwrite_existing
    );
    
    // 乱数の初期化
    int rand_seed = m_setting.value("seed", 12345);
    if (rand_seed < 0) {
        std::random_device rd;
        rand_seed = rd();
    }
    this->mt.seed(rand_seed);

    // ユニットの初期化
    this->configure_units(m_setting);
    // 系の初期化
    this->build_state(c_setting.at("atoms"));
    // ポテンシャル・隣接リストの初期化
    this->build_interaction(c_setting.at("interactions"));

    // その他はシミュレーション毎の設定
}

SimulationRunner::~SimulationRunner() = default;

void SimulationRunner::run() {
    for (const auto& step : j["steps"]) {
        string name = step.at("name");
        std::cout << "シミュレーション: " << name << "を実行します。" << std::endl;

        this->step_dir = parent_dir / name;
        fs::create_directory(step_dir);

        // オブザーバーの初期化
        this->build_observer(step.at("observer"));

        if (step.contains("simulation")) {
            const json& s_setting = step.at("simulation");

            simstate->dt = s_setting.at("dt");

            if (step.value("step", "") == "reset") {
                simstate->current_steps = 0;
            }

            // アンサンブルの初期化
            this->build_ensemble(s_setting.at("ensemble"));
            // シミュレーターの作成
            Simulator simulator(
                *state, *simstate, interaction.get(), integrator.get(), observer.get(), *cell
            );

            // シミュレーションの実行
            simulator.run(s_setting.at("simulation_time"), s_setting.value("use_graph", 100), s_setting.value("log_step", 1000));

        } /*else if (step.contains("minimize")) {
            json mi_setting = step.at("minimize");
            // チェッカーの初期化
            this->build_checker(mi_setting.at("checker"));
            // ミニマイザーの初期化
            this->build_minimizer(mi_setting);

            auto start = std::chrono::steady_clock::now();
            minimizer->run();
            cudaDeviceSynchronize();

            auto end = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(end - start).count();

            std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;
            
        } 
        */ 
        else {
            throw std::runtime_error("stepキーワードが未知です。");
        }

        // シミュレーション終了後の処理
        /*
        if (step.contains("save_last_structure")) {
            json s = step.at("save_last_structure");
            std::string output_path = s.at("path").get<std::string>();
            bool is_unwrap = s.at("is_unwrap").get<bool>();

            md::observers::TrajectoryExporter exporter(*state, output_path, cell.get());
            if (is_unwrap) {
                exporter.export_trajectory_unwrap(*state);
            } else {
                exporter.export_trajectory(*state);
            }
        }
        */
    }
}

void SimulationRunner::configure_units(const json& m_setting) {
    this->unit_type = m_setting.value("unit", "lj");
    if (unit_type == "lj") {
        conversion_factor = 1.0;
        boltzmann_constant = 1.0;
    } else if (unit_type == "metal") {
        boltzmann_constant = 8.617333262145e-5f;
        conversion_factor = 0.964855e-2f;
    } else {
        throw std::runtime_error("未対応のunitです: " + unit_type);
    }
}
