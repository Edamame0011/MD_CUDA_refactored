#include <md/utils/SimulationRunner.hpp>
#include <md/utils/initialize.cuh>

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/observers/Observer.cuh>
#include <md/cells/Cell.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/thermostats/Thermostat.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/convergence_checkers/ConvChecker.cuh>
#include <md/energy_minimizers/EnergyMinimizer.cuh>

#include <md/core/constant.h>
#include <md/core/Simulator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/interactions/LJPotential.cuh>
#include <md/interactions/LJPotential_CLL.cuh>
#include <md/interactions/NNP.cuh>
#include <md/integrators/LangevinIntegrator.cuh>
#include <md/observers/LinearOutput.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/thermostats/NHC1.cuh>
#include <md/thermostats/BussiThermostat.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/observers/LogOutput.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/temperature_schedulers/ConstantScheduler.cuh>
#include <md/temperature_schedulers/LinearScheduler.cuh>
#include <md/interactions/NNP_CSR.cuh>
#include <md/interactions/NNP_aoti.cuh>
#include <md/observers/LinearExportTrajectory.cuh>
#include <md/observers/LogExportTrajectory.cuh>
#include <md/observers/TargetTemperatureExporter.cuh>
#include <md/convergence_checkers/MaxNorm.cuh>
#include <md/energy_minimizers/FireMinimizer.cuh>
#include <md/thermostats/KinEnergyCalculator.cuh>

using namespace md::utils;
using namespace md;

using string = std::string;
using json = nlohmann::json;

SimulationRunner::SimulationRunner(const string& setting_path) {
    // jsonのロード
    std::ifstream f(setting_path);
    if (!f.is_open()) throw std::runtime_error("ファイルを開けません。" );
    this->j = json::parse(f);

    json m_setting = j.at("meta");
    json c_setting = j.at("common_settings");

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
    // セルの初期化
    this->build_cell(c_setting.at("cell"));
    // ポテンシャル・隣接リストの初期化
    this->build_interaction(c_setting.at("interactions"));

    // その他はシミュレーション毎の設定
}

SimulationRunner::~SimulationRunner() = default;

void SimulationRunner::run() {
    for (const auto& step : j["steps"]) {
        string name = step.at("name");
        std::cout << "シミュレーション: " << name << "を実行します。" << std::endl;

        // オブザーバーの初期化
        this->build_observer(step.at("observer"));

        if (step.contains("simulation")) {
            json s_setting = step.at("simulation");

            state->dt = s_setting.at("dt");

            if (j.value("step", "") == "reset") {
                state->current_steps = 0;
            }

            // アンサンブルの初期化
            this->build_ensemble(s_setting.at("ensemble"));
            // シミュレーターの作成
            Simulator simulator(*state, interaction.get(), integrator.get(), observer.get(), cell.get());

                // 時間の計測
            auto start = std::chrono::steady_clock::now();

            // シミュレーションの実行
            simulator.run(s_setting.at("simulation_time"), s_setting.at("use_graph"));
            cudaDeviceSynchronize();

            auto end = std::chrono::steady_clock::now();
            double elapsed_s = std::chrono::duration<double>(end - start).count();

            std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;

        } else if (step.contains("minimize")) {
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
            
        } else {
            throw std::runtime_error("stepキーワードが未知です。");
        }
    }
}

void SimulationRunner::configure_units(const json& m_setting) {
    string unit_type = m_setting.value("unit", "lj");
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
        this->state = md::utils::initialize::generate_binary_lj(n_atoms, density, this->lattice, a_ratio, mt);

    } else if (mode == "from_file") {
        string format = a_setting.value("format", "xyz");
        if (format == "xyz") {
            this->state = md::utils::initialize::read_state_from_xyz(this->lattice, a_setting.at("path"));
        } else {
            throw std::runtime_error("未対応のファイルフォーマットです: " + format);
        }
    } else {
        throw std::runtime_error("未対応のatoms modeです: " + mode);
    }

    state->current_steps = 0;
}

void SimulationRunner::build_cell(const json& c_setting) {
    string c_type = c_setting.value("type", "cubic");

    if (c_type == "cubic") {
        this->cell = std::make_unique<md::cells::CubicCell>(lattice);

    } else {
        throw std::runtime_error("未対応のセルタイプです。: " + c_type);
    }
}

void SimulationRunner::build_observer(const json& o_setting) {
    string o_type = o_setting.value("type", "linear");

    if (o_type == "linear") {
        int interval = o_setting.at("interval").get<int>();

        this->observer = std::make_unique<md::observers::LinearOutput>(
            interval, 
            interaction.get()
        );

    } else if (o_type == "log") {
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        int counter = 5;

        this->observer = std::make_unique<md::observers::LogOutput>(
            log_interval, 
            counter, 
            interaction.get()
        );

    } else if (o_type == "linear_export_trajectory") {
        int interval = o_setting.at("interval").get<int>();
        bool is_unwrap = o_setting.at("is_unwrap").get<bool>();
        string output_path = o_setting.at("output_path").get<string>();

        this->observer = std::make_unique<md::observers::LinearExportTrajectory>(
            interval, 
            is_unwrap, 
            *state, 
            cell.get(), 
            output_path
        );

    } else if (o_type == "log_export_trajectory") {
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        int counter = 5;
        bool is_unwrap = o_setting.at("is_unwrap").get<bool>();
        string output_path = o_setting.at("output_path").get<string>();
        
        this->observer = std::make_unique<md::observers::LogExportTrajectory>(
            log_interval, 
            counter, 
            is_unwrap, 
            *state, 
            cell.get(), 
            output_path
        );

    } else if (o_type == "target_temperature_export") {
        std::vector<float> target_temperatures = o_setting.at("target_temperatures").get<std::vector<float>>();
        float initial_temperature = o_setting.at("initial_temperature").get<float>();
        float cooling_rate_per_step = o_setting.at("cooling_rate_per_step").get<float>();
        string output_path = o_setting.at("output_path").get<string>();
        bool is_unwrap = o_setting.at("is_unwrap").get<bool>();

        this->observer = std::make_unique<md::observers::TargetTemperatureExporter>(
            target_temperatures, 
            initial_temperature, 
            cooling_rate_per_step, 
            output_path, 
            cell.get(), 
            is_unwrap
        );

    } else {
        throw std::runtime_error("未対応のoutput typeです: " + o_type);
    }
}

void SimulationRunner::build_ensemble(const json& e_setting) {
    string ensemble = e_setting.value("type", "NVE");

        if (ensemble == "NVE") {
            md::utils::initialize::init_velocities(*state, e_setting.at("temperature"), mt);
            this->thermostat = std::make_unique<md::thermostats::NoThermostat>();
            this->integrator = std::make_unique<md::integrators::ConstantVolume>(this->thermostat.get());

        } else if (ensemble == "NVT") {
            md::utils::initialize::init_velocities(*state, e_setting.at("temperature"), mt);
            
            // Schedulerの構築
            string sched_type = e_setting.value("scheduler", "constant");
            if (sched_type == "constant") {
                this->scheduler = std::make_unique<md::temperature_schedulers::ConstantScheduler>(e_setting.at("temperature"));

            } else if (sched_type == "linear") {
                float rate_per_step = (float)e_setting.at("rate_per_unit_time") * state->dt;
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
                nhc->init(*state);
                this->thermostat = std::move(nhc);
                this->integrator = std::make_unique<md::integrators::ConstantVolume>(this->thermostat.get());

            } 
            else if (thermo_type == "Bussi") {
                float tau = e_setting.value("tau", 1.0f); 
                int seed = e_setting.value("seed", 12345);
                auto bussi = std::make_unique<md::thermostats::BussiThermostat>(tau, this->scheduler.get());
                bussi->init(*state, seed);
                this->thermostat = std::move(bussi);
                this->integrator = std::make_unique<md::integrators::ConstantVolume>(this->thermostat.get());

            } 
            else if (thermo_type == "Langevin") {
                float gamma = 1.0f / e_setting.value("tau", 1.0f);
                int seed = e_setting.value("seed", 12345);
                auto langevin = std::make_unique<md::integrators::LangevinIntegrator>(gamma, seed, this->scheduler.get());
                langevin->init(*state, seed);
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
    bool use_cll = i_setting.value("cell_list", false);

    if (use_cll) {
        /*
        if (p_type == "lennard_jones") {
            this->interaction = md::utils::initialize::init_LJPotential_CLL_from_json(p_setting, *state, cell.get(), nl.get());
        } else throw std::runtime_error("未対応のpotential typeです: " + p_type);
        */

        throw std::runtime_error("セルリスト法にはまだ未対応です。(実装はありますがSimulationRunner側が対応していないです。そのうち対応します。)");

    } else {
        // neighbour listの初期化
        json n_setting = i_setting.at("neighbour_list");
        float cutoff = n_setting.value("cutoff", 5.0f);
        float margin = n_setting.value("margin", 1.0f);

        this->nl = std::make_unique<NeighbourList>(*state, cutoff, margin);
        nl->generate(*state, cell.get());

        // potantialの初期化
        json p_setting = i_setting.at("potentials");
        string p_type = p_setting.value("type", "lennard_jones");

        if (p_type == "lennard_jones") {
            this->interaction = md::utils::initialize::init_LJPotential_from_json(p_setting, *state, cell.get(), nl.get());
            
        } else if (p_type == "NNP") {
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

        } else throw std::runtime_error("未対応のpotential typeです: " + p_type);
    }
}

void SimulationRunner::build_checker(const json& ch_setting) {
    string type = ch_setting.value("type", "max_norm");

    if (type == "max_norm") {
        float threshold = ch_setting.at("threshold").get<float>();
        this->checker = std::make_unique<md::convergence_checkers::MaxNorm>(threshold);
    } else {
        throw std::runtime_error("未定義のcheckerです。");
    }
}

void SimulationRunner::build_minimizer(const json& mi_setting) {
    string type = mi_setting.value("type", "fire");

    if (type == "fire") {
        auto fire = std::make_unique<md::energy_minimizers::FireMinimizer>(*state, cell.get(), interaction.get(), observer.get(), checker.get());
        if (!(mi_setting.at("params") == "default")) {
            auto p_setting = mi_setting.at("params");
            fire->set_hyper_parameters(
                p_setting.at("n_max"), 
                p_setting.at("n_delay"), 
                p_setting.at("n_neg_max"), 
                p_setting.at("dt_start"), 
                p_setting.at("t_max"), 
                p_setting.at("t_min"), 
                p_setting.at("f_inc"), 
                p_setting.at("f_dec"), 
                p_setting.at("alpha_start"), 
                p_setting.at("f_alpha"), 
                p_setting.at("initialdelay")
            );
        }

        minimizer = std::move(fire);
    } else {
        throw std::runtime_error("未定義のminimizerです。");
    }
}