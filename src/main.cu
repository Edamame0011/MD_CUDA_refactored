#include <md/utils/initialize_from_json.cuh>

#include <external/nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <chrono>
#include <cmath>
#include <iostream>

#include <md/convergence_checkers/ConvChecker.cuh>
#include <md/energy_minimizers/EnergyMinimizer.cuh>
#include <md/energy_minimizers/FireMinimizer.cuh>
#include <md/convergence_checkers/MaxNorm.cuh>

using json = nlohmann::json;

template <typename CellType>
void simulate(CellType& cell, md::State* state, md::Interaction& interaction, std::mt19937& mt, const json& j) {
    json s_setting = j.at("simulation");
    json o_setting = j.at("output");

    state->dt = s_setting.at("dt");
    if (j.value("step", "") == "reset") {
        state->current_steps = 0;
    }
    // アンサンブルの初期化
    auto ensemble = md::utils::initialize::build_ensemble(s_setting.at("ensemble"), *state, mt);

    // オブザーバーの初期化
    auto observer = md::utils::initialize::build_observer(o_setting);

    // simulatorの初期化
    md::Simulator simulator(*state, &interaction, ensemble.integrator.get(), observer.get(), cell);

    // 時間の計測
    auto start = std::chrono::steady_clock::now();

    // シミュレーションの実行
    simulator.run(s_setting.at("simulation_time"), s_setting.at("use_graph"));
    cudaDeviceSynchronize();
    
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;
}

template <typename CellType>
void energy_minimize(CellType& cell, md::State* state, md::Interaction& interaction, std::mt19937& mt, const json& j) {
    json setting = j.at("minimize");
    json o_setting = j.at("output");

    if (j.value("step", "") == "reset") {
        state->current_steps = 0;
    }

    // オブザーバーの初期化
    auto observer = md::utils::initialize::build_observer(o_setting);

    // チェッカーの初期化
    std::unique_ptr<md::ConvChecker> checker = nullptr;
    if (setting.at("checker").at("type") == "max_norm") {
        float threshold = setting.at("checker").at("threshold");
        checker = std::make_unique<md::convergence_checkers::MaxNorm>(threshold);
    }
    else {
        throw std::runtime_error("未定義のcheckerです。");
    }

    // ミニマイザーの初期化
    std::unique_ptr<md::EnergyMinimizer> minimizer = nullptr;
    if (setting.at("type") == "fire") {
        auto fire = std::make_unique<md::energy_minimizers::FireMinimizer<CellType>>(*state, cell, &interaction, observer.get(), checker.get());
        if (!(setting.at("params") == "default")) {
            auto p_setting = setting.at("params");
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
    }
    else {
        throw std::runtime_error("未定義のminimizerです。");
    }

    auto start = std::chrono::steady_clock::now();
    minimizer->run();
    cudaDeviceSynchronize();
    
    auto end = std::chrono::steady_clock::now();
    double elapsed_s = std::chrono::duration<double>(end - start).count();

    std::cout << "かかった時間：" << elapsed_s << "s" << std::endl;
}

template <typename CellType>
void step(md::State* state, const std::array<std::array<float, 3>, 3>& lattice, std::mt19937& mt, const json& j) {
    CellType cell(lattice);
    // 隣接リストとポテンシャルの初期化
    json nl_setting = j.at("common_settings").at("interactions").at("neighbour_list");
    float cutoff = nl_setting.value("cutoff", 5.0f);
    float margin = nl_setting.value("margin", 1.0f);
    if ((bool)nl_setting.value("cell_list", false)) {
        int M = std::max(3, (int)(lattice[0][0] / (cutoff + margin)));
        md::utils::CellList cll(M, cell.Lbox, *state);
        md::utils::NeighbourList_CLL nl(*state, cutoff, margin, cll);
        nl.generate(*state, cell);
        auto interaction = md::utils::initialize::build_interaction(j.at("common_settings").at("interactions").at("potentials"), *state, cell, &nl);
        
        for (const auto& step : j["steps"]) {
            std::string name = step.at("name");
            std::cout << "シミュレーション: " << name << "を実行します。" << std::endl;
            if (step.contains("simulation")) simulate(cell, state, *interaction, mt, step);
            else if (step.contains("minimize")) energy_minimize(cell, state, *interaction, mt, step);
            else std::cout << "未知のキーワードです。" << std::endl;
        }
    } else {
        md::utils::NeighbourList<CellType> nl(*state, cutoff, margin);
        nl.generate(*state, cell);
        auto interaction = md::utils::initialize::build_interaction(j.at("common_settings").at("interactions").at("potentials"), *state, cell, &nl);

        for (const auto& step : j["steps"]) {
            std::string name = step.at("name");
            std::cout << "シミュレーション: " << name << "を実行します。" << std::endl;
            if (step.contains("simulation")) simulate(cell, state, *interaction, mt, step);
            else if (step.contains("minimize")) energy_minimize(cell, state, *interaction, mt, step);
            else std::cout << "未知のキーワードです。" << std::endl;
        }
    }
}

int main(int argc, char* argv[]) {
    try {
        // コマンドライン引数の処理
        if (argc < 2) {
            std::cerr << "jsonファイルのパスを入力してください。" << std::endl;
            return 1;
        }
    
        std::string json_path = argv[1];
    
        // jsonファイルの処理
        std::ifstream f(json_path);
        if (!f.is_open()) throw std::runtime_error("ファイルを開けません。" );
        json j = json::parse(f);

        json meta = j.at("meta");
        json c_setting = j.at("common_settings");
        int rand_seed = meta.value("seed", 12345);
        if (rand_seed < 0) {
            std::random_device rd;
            rand_seed = rd();
        }
        std::mt19937 mt(rand_seed);
        // ユニットの初期化
        md::utils::initialize::configure_units(meta);
        // 系の初期化
        std::array<std::array<float, 3>, 3> lattice;
        auto state = md::utils::initialize::init_state(lattice, c_setting.at("atoms"), mt);
        state->current_steps = 0;
        std::string cell_type = c_setting.at("cell").value("type", "cubic");
        if (cell_type == "cubic") {
            step<md::cells::CubicCell>(state.get(), lattice, mt, j);
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