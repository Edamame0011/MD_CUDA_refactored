#include <md/core/Initializer.cuh>
#include <md/utils/compute.cuh>
#include <md/core/constant.h>

#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

#include <thrust/host_vector.h>

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

using namespace md;

void Initializer::read_state_from_xyz(State& state, const std::string& path) {
    std::ifstream file(path);

    if(!file.is_open()) {
        std::cerr << "構造ファイルを開けません。" << std::endl;
        throw std::runtime_error("構造ファイルを開けません: " + path);
    }

    std::string line;
    std::getline(file, line);
    state.n_atoms = std::stoi(line);
    int num_atoms = state.n_atoms;
    state.d_box.x.resize(num_atoms, 0);
    state.d_box.y.resize(num_atoms, 0);
    state.d_box.z.resize(num_atoms, 0);
    state.d_velocities.x.resize(num_atoms, 0);
    state.d_velocities.y.resize(num_atoms, 0);
    state.d_velocities.z.resize(num_atoms, 0);

    std::getline(file, line);

    // 原子の情報を保持する変数
    thrust::host_vector<int64_t> h_atomic_numbers(num_atoms);
    thrust::host_vector<float> h_x(num_atoms);
    thrust::host_vector<float> h_y(num_atoms);
    thrust::host_vector<float> h_z(num_atoms);
    thrust::host_vector<float> h_force_x(num_atoms);
    thrust::host_vector<float> h_force_y(num_atoms);
    thrust::host_vector<float> h_force_z(num_atoms);
    thrust::host_vector<float> h_masses(num_atoms);

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
    state.d_positions.x = h_x;
    state.d_positions.y = h_y;
    state.d_positions.z = h_z;
    state.d_forces.x = h_force_x;
    state.d_forces.y = h_force_y;
    state.d_forces.z = h_force_z;
    state.d_masses = h_masses;
    state.d_atomic_numbers = h_atomic_numbers;

    std::cout << "構造ファイルを読み込みました：" << path << std::endl;
    std::cout << "原子数：" << num_atoms << std::endl;
}

std::array<std::array<float, 3>, 3> Initializer::find_lattice_from_xyz(const std::string& path) {
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

void Initializer::init_velocities(State& state, float temperature, std::mt19937& mt) {
    // デバイスから質量を転送
    thrust::host_vector<float> masses = state.d_masses;
    // 平均0、分散1のガウス分布
    std::normal_distribution<float> dist_trans(0.0, 1);
    int N = state.n_atoms;

    // ホスト側の配列
    thrust::host_vector<float> h_vel_x(N);
    thrust::host_vector<float> h_vel_y(N);
    thrust::host_vector<float> h_vel_z(N);

    for (int i = 0; i < state.n_atoms; i ++) {
        // 分散を調節
        h_vel_x[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / masses[i]);
        h_vel_y[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / masses[i]);
        h_vel_z[i] = dist_trans(mt) * std::sqrt((boltzmann_constant * temperature * conversion_factor) / masses[i]);
    }

    // デバイスに速度を送信
    state.d_velocities.x = h_vel_x;
    state.d_velocities.y = h_vel_y;
    state.d_velocities.z = h_vel_z;

    // 全体速度の除去
    md::utils::compute::remove_drift(state);
}