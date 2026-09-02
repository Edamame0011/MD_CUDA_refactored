#include <md/utils/SimulationRunner.hpp>
#include <md/utils/initialize.hpp>

#include <md/core/State.hpp>
#include <md/integrators/Integrator.hpp>
#include <md/interactions/Interaction.hpp>
#include <md/observers/Observer.hpp>
#include <md/core/Cell.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.hpp>
#include <md/thermostats/Thermostat.cuh>
#include <md/neighbour/NeighbourList.hpp>
// #include <md/convergence_checkers/ConvChecker.hpp>
// #include <md/energy_minimizers/EnergyMinimizer.hpp>

#include <md/core/constant.h>
#include <md/core/Simulator.hpp>
#include <md/integrators/ConstantVolumeLJ.hpp>
#include <md/interactions/LJPotential.hpp>
// #include <md/interactions/NNP.hpp>
#include <md/integrators/LangevinIntegratorLJ.cuh>
#include <md/observers/LinearEnergiesObserver.hpp>
#include <md/observers/LogEnergiesObserver.hpp>
#include <md/thermostats/NoThermostat.hpp>
#include <md/thermostats/NHC1.hpp>
#include <md/thermostats/BussiThermostat.cuh>
#include <md/temperature_schedulers/ConstantScheduler.hpp>
#include <md/temperature_schedulers/LinearScheduler.hpp>
// #include <md/observers/LinearExportTrajectory.hpp>
// #include <md/observers/LogExportTrajectory.hpp>
// #include <md/observers/TargetTemperatureExporter.hpp>
// #include <md/observers/LogplusStrideExportTrajectory.hpp>
// #include <md/convergence_checkers/MaxNorm.hpp>
// #include <md/energy_minimizers/FireMinimizer.hpp>
// #include <md/thermostats/KinEnergyCalculator.hpp>
#include <md/observers/TrajectoryExporter.hpp>
#include <md/neighbour/CellList.hpp>
#include <md/neighbour/NeighbourList.hpp>
#include <md/neighbour/NativeNeighbourList.hpp>
#include <md/neighbour/CellListNeighbourList.hpp>

#include <cmath>
#include <fstream>
#include <iostream>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <vector>

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

            // LinearScheduler reads the host-side current step while a graph is
            // captured.  Re-capture one MD step at a time so its temperature is
            // refreshed; constant-temperature and NVE runs keep the fast path.
            const float simulation_time = s_setting.at("simulation_time").get<float>();
            const int log_step = s_setting.value("log_step", 1000);
            if (dynamic_cast<md::temperature_schedulers::LinearScheduler*>(scheduler.get()) != nullptr) {
                const int steps = static_cast<int>(simulation_time / simstate->dt);
                for (int i = 0; i < steps; ++i) {
                    simulator.run(simstate->dt, 1, log_step);
                }
            } else {
                simulator.run(simulation_time, s_setting.value("use_graph", 100), log_step);
            }

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
        if (step.contains("save_last_structure")) {
            const json& s = step.at("save_last_structure");
            fs::path output_path = step_dir / s.value("path", "last_structure.xyz");
            bool is_unwrap = s.value("is_unwrap", false);

            md::observers::TrajectoryExporter exporter(*state, output_path.string(), cell.get());
            if (is_unwrap) {
                exporter.export_trajectory_unwrap(*state);
            } else {
                exporter.export_trajectory(*state);
            }
        }
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

void SimulationRunner::build_state(const json& a_setting) {
    string mode = a_setting.value("mode", "");
    
    if (mode == "generate_binary_lj") {
        int n_atoms = a_setting.at("n_atoms").get<int>();
        float density = a_setting.at("density").get<float>();
        auto ratio_vec = a_setting.at("ratio").get<std::vector<float>>();
        if (ratio_vec.size() < 2) throw std::runtime_error("ratioには少なくとも2つの要素が必要です。");
        
        float a_ratio = ratio_vec[0] / (ratio_vec[0] + ratio_vec[1]);
        this->state = md::utils::generate_binary_lj(n_atoms, density, this->cell, a_ratio, mt);

        // State currently leaves trajectory bookkeeping arrays uninitialized.
        // Initialize them here so save_last_structure is valid before any sorting.
        cudaMemset(state->image.x, 0, n_atoms * sizeof(int));
        cudaMemset(state->image.y, 0, n_atoms * sizeof(int));
        cudaMemset(state->image.z, 0, n_atoms * sizeof(int));
        thrust::sequence(thrust::device, state->particle_id, state->particle_id + n_atoms);

    } /*else if (mode == "from_file") {
        string format = a_setting.value("format", "xyz");
        if (format == "xyz") {
            this->state = md::utils::initialize::read_state_from_xyz(this->cell, a_setting.at("path"));
        } else {
            throw std::runtime_error("未対応のファイルフォーマットです: " + format);
        }
    } else {
        throw std::runtime_error("未対応のatoms modeです: " + mode);
    }
*/
    this->simstate = std::make_unique<md::SimState>();
}

void SimulationRunner::build_observer(const json& o_setting) {
    string o_type = o_setting.value("type", "linear");

    if (o_type == "linear") {
        int interval = o_setting.at("interval").get<int>();
        fs::path output_path = step_dir / o_setting.value("output_path", "observer_output");

        this->observer = std::make_unique<md::observers::LinearEnergiesObserver>(
            interval, 
            interaction.get(), 
            output_path.string()
        );

    } else if (o_type == "log") {
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        int counter = 5;
        fs::path output_path = step_dir / o_setting.value("output_path", "observer_output.txt");

        this->observer = std::make_unique<md::observers::LogEnergiesObserver>(
            log_interval, 
            counter, 
            interaction.get(), 
            output_path.string()
        );

/*
    } else if (o_type == "linear_export_trajectory") {
        int interval = o_setting.at("interval").get<int>();
        bool is_unwrap = o_setting.at("is_unwrap").get<bool>();
        fs::path output_path = step_dir / o_setting.value("output_path", "observer_output.xyz");

        this->observer = std::make_unique<md::observers::LinearExportTrajectory>(
            interval, 
            is_unwrap, 
            *state, 
            cell.get(), 
            output_path.string()
        );

    } else if (o_type == "log_export_trajectory") {
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        int counter = 5;
        bool is_unwrap = o_setting.at("is_unwrap").get<bool>();
        fs::path output_path = step_dir / o_setting.value("output_path", "observer_output.xyz");
        fs::path temp_path = step_dir / o_setting.value("temp_path", "observer_temp.txt");
        
        this->observer = std::make_unique<md::observers::LogExportTrajectory>(
            log_interval, 
            counter, 
            is_unwrap, 
            *state, 
            cell.get(), 
            output_path.string(), 
            temp_path.string()
        );

    } else if (o_type == "target_temperature_export") {
        std::vector<float> target_temperatures = o_setting.at("target_temperatures").get<std::vector<float>>();
        float initial_temperature = o_setting.at("initial_temperature").get<float>();
        float cooling_rate_per_step = o_setting.at("cooling_rate_per_step").get<float>();
        fs::path output_path = step_dir / o_setting.at("output_path").get<string>();
        bool is_unwrap = o_setting.at("is_unwrap").get<bool>();

        this->observer = std::make_unique<md::observers::TargetTemperatureExporter>(
            target_temperatures, 
            initial_temperature, 
            cooling_rate_per_step, 
            output_path.string(), 
            cell.get(), 
            is_unwrap
        );


    } else if(o_type == "log_plus_stride_export_trajectory") {
        size_t num_trajectory = o_setting.at("num_trajectory");
        float stride = o_setting.at("stride");
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        int counter = 5;
        bool is_unwrap = o_setting.at("is_unwrap").get<bool>();
        fs::path output_path = step_dir / o_setting.at("output_path").get<string>();
        
        this->observer = std::make_unique<md::observers::LogplusStrideExportTrajectory>(
            num_trajectory, 
            stride, 
            log_interval, 
            counter, 
            is_unwrap, 
            *state, 
            cell.get(), 
            output_path.string()
        );
*/

    } else {
        throw std::runtime_error("未対応のoutput typeです: " + o_type);
    }
}


void SimulationRunner::build_ensemble(const json& e_setting) {
    string ensemble = e_setting.value("type", "NVE");

    // Destroy dependants before replacing the scheduler they refer to.
    this->integrator.reset();
    this->thermostat.reset();
    this->scheduler.reset();

    if (ensemble == "NVE") {
        md::utils::init_velocities(*state, e_setting.at("temperature"), mt);
        this->thermostat = std::make_unique<md::thermostats::NoThermostat>();
        this->integrator = std::make_unique<md::integrators::ConstantVolumeLJ>(this->thermostat.get());

    } else if (ensemble == "NVT") {
        md::utils::init_velocities(*state, e_setting.at("temperature"), mt);
        
        // Schedulerの構築
        string sched_type = e_setting.value("scheduler", "constant");
        if (sched_type == "constant") {
            this->scheduler = std::make_unique<md::temperature_schedulers::ConstantScheduler>(e_setting.at("temperature"));

        } else if (sched_type == "linear") {
            float rate_per_step = e_setting.at("rate_per_unit_time").get<float>() * simstate->dt;
            this->scheduler = std::make_unique<md::temperature_schedulers::LinearScheduler>(e_setting.at("temperature"), rate_per_step);

        } else {
            throw std::runtime_error("未対応のschedulerです: " + sched_type);
        }

        // Thermostatの構築
        string thermo_type = e_setting.value("thermostat", "Nose-Hoover");
        if (thermo_type == "Nose-Hoover") {
            auto nhc = std::make_unique<md::thermostats::NHC1>(
                e_setting.value("tau", 1.0f), this->scheduler.get()
            );
            nhc->init(*state, *simstate);
            this->thermostat = std::move(nhc);
            this->integrator = std::make_unique<md::integrators::ConstantVolumeLJ>(this->thermostat.get());

        } 
        else if (thermo_type == "Bussi") {
            float tau = e_setting.value("tau", 1.0f); 
            unsigned long long seed = e_setting.value("seed", 12345ULL);
            auto bussi = std::make_unique<md::thermostats::BussiThermostat>(tau, this->scheduler.get());
            bussi->init(*state, *simstate, seed);
            this->thermostat = std::move(bussi);
            this->integrator = std::make_unique<md::integrators::ConstantVolumeLJ>(this->thermostat.get());

        } 
        else if (thermo_type == "Langevin") {
            float gamma = 1.0f / e_setting.value("tau", 1.0f);
            unsigned long long seed = e_setting.value("seed", 12345ULL);
            auto langevin = std::make_unique<md::integrators::LangevinIntegratorLJ>(gamma, seed, this->scheduler.get());
            langevin->init(*state, *simstate, seed);
            this->integrator = std::move(langevin);

        }
        else {
            throw std::runtime_error("未対応のthermostatです: " + thermo_type);
        }
    
    } else {
        throw std::runtime_error("未対応のensembleです: " + ensemble);
    }
}

void SimulationRunner::build_interaction(const json& i_setting) {
    use_cell_list = i_setting.value("cell_list", false);

    json n_setting = i_setting.at("neighbour_list");
    float cutoff = n_setting.value("cutoff", 5.0f);
    float margin = n_setting.value("margin", 1.0f);
    int max_neighbours = n_setting.value("max_neighbours", 100);

    if (use_cell_list) {
        auto lattice = cell->get_lattice();
        int Mx = std::max(3, (int)(lattice[0] / (cutoff + margin)));
        int My = std::max(3, (int)(lattice[1] / (cutoff + margin)));
        int Mz = std::max(3, (int)(lattice[2] / (cutoff + margin)));
        std::array<int, 3> M = {Mx, My, Mz};

        // cell listの初期化
        this->cl = std::make_unique<CellList>(M, *state, *cell);

        // neighbour listの初期化
        this->nl = std::make_unique<md::neighbour::CellListNeighbourList>(state->n_atoms, max_neighbours, cutoff, margin, *cl);
        nl->generate(*state, *simstate, *cell);

    } else {
        // neighbour listの初期化
        this->nl = std::make_unique<md::neighbour::NativeNeighbourList>(state->n_atoms, max_neighbours, cutoff, margin);
        nl->generate(*state, *simstate, *cell);
    }

    // potantialの初期化
    json p_setting = i_setting.at("potentials");
    string p_type = p_setting.value("type", "lennard_jones");

    if (p_type == "lennard_jones") {
        std::vector<float> sigma = p_setting.at("sigma").get<std::vector<float>>();
        std::vector<float> epsilon = p_setting.at("epsilon").get<std::vector<float>>();
        std::vector<float> cutoff = p_setting.at("cutoff").get<std::vector<float>>();
        std::vector<int> numbers = p_setting.at("numbers").get<std::vector<int>>();
        int num_species = numbers.size();

        this->interaction = std::make_unique<md::interactions::LJPotential>(num_species, *cell, nl.get(), sigma, epsilon, cutoff);

    } /* else if (p_type == "NNP") {
        float cutoff = p_setting.at("cutoff").get<float>();
        int max_edges = p_setting.at("max_edges").get<int>();
        string model_path = p_setting.at("model_path").get<string>();
        
        this->interaction = std::make_unique<md::interactions::NNP>(
            *state, 
            cell.get(), 
            nl.get(), 
            cutoff, 
            max_edges, 
            model_path
        );

    } else if (p_type == "NNP_csr") {
        float cutoff = p_setting.at("cutoff").get<float>();
        int max_edges = p_setting.at("max_edges").get<int>();
        string model_path = p_setting.at("model_path").get<string>();
    
        this->interaction =  std::make_unique<md::interactions::NNP_CSR>(
            *state, 
            cell.get(), 
            nl.get(), 
            cutoff, 
            max_edges, 
            model_path
        );

    } else if (p_type == "NNP_aoti") {
        float cutoff = p_setting.at("cutoff").get<float>();
        int max_edges = p_setting.at("max_edges").get<int>();
        string model_path = p_setting.at("model_path").get<string>();

        this->interaction =  std::make_unique<md::interactions::NNP_aoti>(
            *state, 
            cell.get(), 
            nl.get(), 
            cutoff, 
            max_edges, 
            model_path
        );

    } else if (p_type == "NNP_force_aoti") {
        float cutoff = p_setting.at("cutoff").get<float>();
        int max_edges = p_setting.at("max_edges").get<int>();
        string force_model_path = p_setting.at("force_model_path").get<string>();
        string energy_model_path = p_setting.at("energy_model_path").get<string>();

        this->interaction =  std::make_unique<md::interactions::NNP_force_aoti>(
            *state, 
            cell.get(), 
            nl.get(), 
            cutoff, 
            max_edges, 
            force_model_path, 
            energy_model_path
        );

    } */ else throw std::runtime_error("未対応のpotential typeです: " + p_type);
}
