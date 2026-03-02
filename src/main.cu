#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/core/Simulator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/interactions/NNPTorchScript.cuh>
#include <md/interactions/LJPotential.cuh>
#include <md/observers/LinearOutput.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/initialize.cuh>
#include <md/observers/LogOutput.cuh>

#include <external/nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>

using json = nlohmann::json;

template <typename CellType>
void simulate(CellType& cell, const json& j) {
    std::mt19937 mt(123456789);

    json m_setting = j.at("meta");
    json a_setting = j.at("atoms");
    json s_setting = j.at("simulation");
    json o_setting = j.at("output");
    json i_setting = j.at("interactions");

    md::State state;
    std::array<std::array<float, 3>, 3> lattice = {};
    
    // その他の設定
    state.dt = s_setting.at("dt");
    std::string unit_type = m_setting.value("unit", "lj");
    if(unit_type == "lj") {
        md::conversion_factor = 1.0;
        md::boltzmann_constant = 1.0;
    }
    else if(unit_type == "metal") {
        md::boltzmann_constant = 8.617333262145e-5f;
        md::conversion_factor = 0.964855e-2f;
    }
    else {
        throw std::runtime_error("未対応のunitです: " + unit_type);
    }

    // 系の初期化
    std::string mode = a_setting.value("mode", "");
    if (mode == "generate_binary_lj") {
        int n_atoms = a_setting.at("n_atoms").get<int>();
        float density = a_setting.at("density").get<float>();
        auto ratio_vec = a_setting.at("ratio").get<std::vector<float>>();
        if (ratio_vec.size() < 2) {
            throw std::runtime_error("ratioには少なくとも2つの要素が必要です。");
        }
        float a_ratio = ratio_vec[0] / (ratio_vec[0] + ratio_vec[1]);
    
        md::utils::initialize::generate_binary_lj(state, n_atoms, density, lattice, a_ratio, mt);
    }
    else if (mode == "from_file") {
        std::string data_path = a_setting.at("path");
        std::string format = a_setting.value("format", "xyz");
        if(format == "xyz") {
            lattice = md::utils::initialize::read_state_from_xyz(state, data_path);
        }
        else {
            throw std::runtime_error("未対応のファイルフォーマットです: " + format);
        }
    }
    else {
        throw std::runtime_error("未対応のatoms mode、またはmodeが指定されていません: " + mode);
    }

    // セルの初期化
    cell.init(lattice[0][0]);

    // 隣接リストの作成
    json nl_setting = i_setting.at("neighbour_list");
    md::utils::NeighbourList nl(nl_setting.at("cutoff"), nl_setting.at("margin"));
    nl.generate(state, cell);

    // ポテンシャルの初期化
    json p_setting = i_setting.at("potentials");
    std::string p_type = p_setting.at("type");
    std::unique_ptr<md::Interaction> interaction;
    if (p_type == "lennard_jones") {
        auto pot = md::utils::initialize::init_LJPotential_from_json(p_setting, state, cell, &nl);
        interaction = std::make_unique<md::interactions::LJPotential<CellType>>(std::move(pot));
    }
    else if (p_type == "NNP_TorchScript") {
        interaction = std::make_unique<md::interactions::NNPTorchScript<CellType>>(state, cell, &nl, p_setting.at("model_path"));
    }
    else {
        throw std::runtime_error("未対応のpotential typeです: " + p_type);
    }

    // observerの初期化
    std::unique_ptr<md::Observer> observer;
    std::string o_type = o_setting.value("type", "linear");
    if (o_type == "linear") {
        observer = std::make_unique<md::observers::LinearOutput>(o_setting.at("output_interval"));
    }
    if (o_type == "log") {
        int divisions = o_setting.at("divisions");
        float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
        observer = std::make_unique<md::observers::LogOutput>(log_interval, 5);
    }
    else {
        throw std::runtime_error("未対応のoutput typeです: " + o_type);
    }

    // 熱浴・integratorの初期化
    std::unique_ptr<md::Thermostat> thermostat;
    std::unique_ptr<md::Integrator> integrator;
    json e_setting = s_setting.at("ensemble");
    std::string ensemble = e_setting.value("type", "NVE");
    if (ensemble == "NVE") {
        md::utils::initialize::init_velocities(state, e_setting.at("temperature"), mt);
        thermostat = std::make_unique<md::thermostats::NoThermostat>();
        integrator = std::make_unique<md::integrators::ConstantVolume>(thermostat.get());
    }
    else {
        throw std::runtime_error("未対応のensembleです: " + ensemble);
    }

    // simulatorの初期化
    md::Simulator simulator(state, interaction.get(), integrator.get(), observer.get(), cell);

    // 時間の計測
    auto start = std::chrono::steady_clock::now();

    // シミュレーションの実行
    simulator.run(s_setting.at("simulation_time"));
    
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // コマンドライン引数の処理
        if (argc < 2) {
            std::cout << "jsonファイルのパスを入力してください。" << std::endl;
            return 1;
        }
    
        std::string json_path = argv[1];
    
        // jsonファイルの処理
        std::ifstream f(json_path);
        if (!f.is_open()) throw std::runtime_error("ファイルを開けません。" );
        json j = json::parse(f);
    
        json s_setting = j.at("simulation");
    
        std::string cell_type = s_setting.value("cell_type", "cubic");
        if (cell_type == "cubic") {
            md::cells::CubicCell cell;
            simulate(cell, j);
        }
        else {
            throw std::runtime_error("未対応のcell typeです: " + cell_type);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "[Error]" << e.what() << std::endl;
        return 1;
    }
    return 0;
}