#include <md/utils/initialize.cuh>

#include <md/utils/compute.cuh>
#include <md/core/constant.h>
#include <md/core/State.cuh>

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <algorithm>

namespace {
    //文字列からLatticeを見つける
    std::string find_lattice(std::string input) {
        //開始位置のキーワード
        std::string start_tag = "Lattice=\"";

        //開始位置
        std::size_t start_position = input.find(start_tag);

        if(start_position != std::string::npos){
            start_position = start_position + start_tag.length();

            //開始位置から次の"を探す
            std::size_t end_position = input.find('"', start_position);

            if(end_position != std::string::npos){
                //抜き出す部分の長さ
                std::size_t length = end_position - start_position;

                //文字列の抜き出し
                std::string result = input.substr(start_position, length);

                return result;
            }

            else{ 
                throw std::runtime_error("終了のダブルクオーテーションが見つかりません。"); 
            }
        }

        else{ 
            throw std::runtime_error("ファイルにLatticeデータが含まれていません。"); 
        }
    }
    //原子種類と原子番号を関連づけるmap
    const std::map<std::string, int> atom_number_map = {
            {"H",  1},
            {"He", 2},
            {"Li", 3},
            {"Be", 4},
            {"B",  5},
            {"C",  6},
            {"N",  7},
            {"O",  8},
            {"F",  9},
            {"Ne", 10},
            {"Na", 11},
            {"Mg", 12},
            {"Al", 13},
            {"Si", 14},
            {"P",  15},
            {"S",  16},
            {"Cl", 17},
            {"Ar", 18},
            {"K",  19},
            {"Ca", 20}
        };

    //原子種類と原子質量を関連づけるmap
    const std::map<std::string, double> atom_mass_map = {
            {"H",   1.0080},
            {"He",  4.0026},
            {"Li",  6.94},
            {"Be",  9.0122},
            {"B",   10.81},
            {"C",   12.011},
            {"N",   14.007},
            {"O",   15.999},
            {"F",   18.998},
            {"Ne",  20.180},
            {"Na",  22.990},
            {"Mg",  24.305},
            {"Al",  26.982},
            {"Si",  28.0855},
            {"P",   30.974},
            {"S",   32.06},
            {"Cl",  35.45},
            {"Ar",  39.95},
            {"K",   39.098},
            {"Ca",  40.078}
        };
}

using namespace md::utils;

std::unique_ptr<md::State> initialize::read_state_from_xyz(std::array<std::array<float, 3>, 3>& lattice, const std::string& path) {
    std::ifstream file(path);

    if(!file.is_open()) {
        std::cerr << "構造ファイルを開けません。" << std::endl;
        throw std::runtime_error("構造ファイルを開けません: " + path);
    }

    std::string line;
    std::getline(file, line);
    int N = std::stoi(line);

    auto state = std::make_unique<md::State>(N);

    // latticeの情報を取得
    std::getline(file, line);
    std::array<float, 3> lattice_x, lattice_y, lattice_z;
    // latticeの部分を読み込む
    // コメントからlatticeの部分を抜き出す
    std::string lattice_comment = find_lattice(line);
    // 文字列をストリームに変換
    std::istringstream iss(lattice_comment);
    iss >> lattice_x[0] >> lattice_x[1] >> lattice_x[2] >> 
           lattice_y[0] >> lattice_y[1] >> lattice_y[2] >>
           lattice_z[0] >> lattice_z[1] >> lattice_z[2];
    lattice = {lattice_x, lattice_y, lattice_z};

    // 原子の情報を保持する変数
    std::vector<int> h_atomic_numbers(N);
    std::vector<float> h_x(N);
    std::vector<float> h_y(N);
    std::vector<float> h_z(N);
    std::vector<float> h_vel_x(N);
    std::vector<float> h_vel_y(N);
    std::vector<float> h_vel_z(N);
    std::vector<float> h_force_x(N);
    std::vector<float> h_force_y(N);
    std::vector<float> h_force_z(N);
    std::vector<float> h_masses(N);

    int i = 0;

    while(std::getline(file, line)) {
        std::string atom_type;

        std::istringstream iss(line);

        iss >> atom_type >> h_x[i] >> h_y[i] >> h_z[i] >> h_force_x[i] >> h_force_y[i] >> h_force_z[i];
        
        h_atomic_numbers[i] = atom_number_map.at(atom_type);
        h_masses[i] = atom_mass_map.at(atom_type);

        i ++;
    }

    // デバイスに転送
    state->copy(
        h_x.data(), h_y.data(), h_z.data(), 
        h_vel_x.data(), h_vel_y.data(), h_vel_z.data(), 
        h_force_x.data(), h_force_y.data(), h_force_z.data(), 
        h_masses.data(), h_atomic_numbers.data()
    );

    std::cout << "構造ファイルを読み込みました：" << path << std::endl;
    std::cout << "原子数：" << N << std::endl;

    return state;
}

std::array<std::array<float, 3>, 3> initialize::find_lattice_from_xyz(const std::string& path) {
    std::ifstream file(path);

    if(!file.is_open()) {
        std::cerr << "構造ファイルを開けません。" << std::endl;
        throw std::runtime_error("構造ファイルを開けません: " + path);
    }

    std::string line;
    // 1行目は原子数なためスルー
    std::getline(file, line);

    std::getline(file, line);
    std::array<float, 3> lattice_x, lattice_y, lattice_z;
    // latticeの部分を読み込む
    // コメントからlatticeの部分を抜き出す
    std::string lattice_comment = find_lattice(line);
    // 文字列をストリームに変換
    std::istringstream iss(lattice_comment);
    iss >> lattice_x[0] >> lattice_x[1] >> lattice_x[2] >> 
           lattice_y[0] >> lattice_y[1] >> lattice_y[2] >>
           lattice_z[0] >> lattice_z[1] >> lattice_z[2];
    std::array<std::array<float, 3>, 3> lattice = {lattice_x, lattice_y, lattice_z};
    return lattice;
}

void initialize::init_velocities(State& state, float temperature, std::mt19937& mt) {
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