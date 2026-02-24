#ifndef LJ_POTENTIAL_CUH
#define LJ_POTENTIAL_CUH

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/utils/NeighbourList.cuh>
#include <thrust/device_vector.h>

#include <external/nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <string>

namespace {
    __host__ __device__ float LJpotential(const float rij1, const float sij1) {
        const float rij2 = rij1 * rij1;
        const float rij6 = rij2 * rij2 * rij2;
        const float sij2 = sij1 * sij1;
        const float sij6 = sij2 * sij2;

        return 4.0f * sij6 * (sij6 - rij6) / (rij6 * rij6);
    }

    __host__ __device__ float deriv_1st_LJpotential(const float rij1, const float sij1) {
        const float rij2 = rij1 * rij1;
        const float rij6 = rij2 * rij2 * rij2;
        const float sij2 = sij1 * sij1;
        const float sij6 = sij2 * sij2;

        return -24.0f / rij1 * sij6 * (2.0f * sij6 - rij6) / (rij6 * rij6);
    }

    template <typename CellType>
    struct CalcForce {
        const float *d_x, *d_y, *d_z;
        float *d_force_x, *d_force_y, *d_force_z;
        const float *sigma, *epsilon, *cutoff;
        const int *identifier;
        int num_species, num_atoms;
        CellType cell;

        CalcForce(
            const float *_d_x, 
            const float *_d_y, 
            const float *_d_z, 
            float *_d_force_x, 
            float *_d_force_y, 
            float *_d_force_z, 
            const float *_sigma, 
            const float *_epsilon, 
            const float *_cutoff, 
            const int *_identifier, 
            int _num_atoms, 
            int _num_species, 
            CellType _cell
        ) : 
        d_x(_d_x), 
        d_y(_d_y), 
        d_z(_d_z), 
        d_force_x(_d_force_x), 
        d_force_y(_d_force_y), 
        d_force_z(_d_force_z), 
        sigma(_sigma), 
        epsilon(_epsilon), 
        cutoff(_cutoff), 
        identifier(_identifier), 
        num_atoms(_num_atoms), 
        num_species(_num_species), 
        cell(_cell) {}

        __host__ __device__ void operator() (int idx) {
            int i = idx / num_atoms;
            int j = idx % num_atoms;

            if (j >= i) return;

            // 原子の種類
            const int si = identifier[i];
            const int sj = identifier[j];

            // 距離の計算
            const float x1 = d_x[i];
            const float y1 = d_y[i];
            const float z1 = d_z[i];

            const float x2 = d_x[j];
            const float y2 = d_y[j];
            const float z2 = d_z[j];

            const float dx = x1 - x2;
            const float dy = y1 - y2;
            const float dz = z1 - z2;

            cell.apply_pbc(dx, dy, dz);

            const float dist_sq = dx * dx + dy * dy + dz * dz;

            const float r_c = cutoff[si * num_species + sj];

            // カットオフ距離内かを判定
            if (dist_sq >= r_c * r_c) return;

            // 力の計算
            const float rij1 = sqrtf(dist_sq);
            const float sij1 = sigma[si * num_species + sj];

            float deriv_1st = deriv_1st_LJpotential(rij1, sij1);
            deriv_1st *= epsilon[si * num_species + sj];

            atomicAdd(&d_force_x[i], -deriv_1st * dx / rij1);
            atomicAdd(&d_force_y[i], -deriv_1st * dy / rij1);
            atomicAdd(&d_force_z[i], -deriv_1st * dz / rij1);
    
            atomicAdd(&d_force_x[j], deriv_1st * dx / rij1);
            atomicAdd(&d_force_y[j], deriv_1st * dy / rij1);
            atomicAdd(&d_force_z[j], deriv_1st * dz / rij1);
        }
    };

    template <typename CellType>
    struct CalcPotential {
        const float *d_x, *d_y, *d_z;
        const float *sigma, *epsilon, *cutoff;
        const int *identifier;
        int num_species, num_atoms;
        CellType cell;

        CalcPotential(
            const float *_d_x, 
            const float *_d_y, 
            const float *_d_z, 
            const float *_sigma, 
            const float *_epsilon, 
            const float *_cutoff, 
            const int *_identifier,  
            int _num_atoms, 
            int _num_species, 
            CellType _cell
        ) : 
        d_x(_d_x), 
        d_y(_d_y), 
        d_z(_d_z), 
        sigma(_sigma), 
        epsilon(_epsilon), 
        cutoff(_cutoff), 
        identifier(_identifier),
        num_atoms(_num_atoms), 
        num_species(_num_species), 
        cell(_cell) {}

        __host__ __device__ float operator() (int idx) {
            int i = idx / num_atoms;
            int j = idx % num_atoms;

            if (j <= i) return 0.0f;

            // 原子の種類
            const int si = identifier[i];
            const int sj = identifier[j];

            // 距離の計算
            const float x1 = d_x[i];
            const float y1 = d_y[i];
            const float z1 = d_z[i];

            const float x2 = d_x[j];
            const float y2 = d_y[j];
            const float z2 = d_z[j];

            const float dx = x1 - x2;
            const float dy = y1 - y2;
            const float dz = z1 - z2;

            cell.apply_pbc(dx, dy, dz);

            const float dist_sq = dx * dx + dy * dy + dz * dz;

            const float r_c = cutoff[si * num_species + sj];

            // カットオフ距離内かを判定
            if (dist_sq >= r_c * r_c) return 0.0f;

            // ポテンシャルの計算
            const float rij1 = sqrtf(dist_sq);
            const float sij1 = sigma[si * num_species + sj];

            return LJpotential(rij1, sij1) - LJpotential(r_c, sij1) - deriv_1st_LJpotential(r_c, sij1) * (rij1 - r_c);
        }
    };
}

namespace md::interactions {
    template <typename CellType>
    class LJPotential : public Interaction {
        public: 
            LJPotential(
                int _num_species, 
                CellType _cell, 
                md::utils::NeighbourList *_NL, 
                std::vector<float> _sigma, 
                std::vector<float> _epsilon, 
                std::vector<float> _cutoff, 
                std::vector<int> _identifier
            ) : 
            num_species(_num_species), cutoff(_cutoff), cell(_cell), NL(_NL) {
                // データの転送
                sigma = _sigma;
                epsilon = _epsilon;
                identifier = _identifier;
            }

            void forward(State& state) override {                
                // 力の計算
                thrust::for_each(
                    NL->d_valid_indices().begin(), 
                    NL->d_valid_indices().end(), 
                    CalcForce(
                        thrust::raw_pointer_cast(state.d_positions.x.data()), 
                        thrust::raw_pointer_cast(state.d_positions.y.data()), 
                        thrust::raw_pointer_cast(state.d_positions.z.data()), 
                        thrust::raw_pointer_cast(state.d_forces.x.data()), 
                        thrust::raw_pointer_cast(state.d_forces.y.data()), 
                        thrust::raw_pointer_cast(state.d_forces.z.data()), 
                        thrust::raw_pointer_cast(sigma.data()), 
                        thrust::raw_pointer_cast(epsilon.data()), 
                        thrust::raw_pointer_cast(cutoff.data()), 
                        thrust::raw_pointer_cast(identifier.data()), 
                        state.n_atoms, 
                        num_species, 
                        cell
                    )
                );

                // ポテンシャルの計算
                state.potential_energy = thrust::transform_reduce(
                    NL->d_valid_indices().begin(), 
                    NL->d_valid_indices().end(), 
                    CalcPotential(
                        thrust::raw_pointer_cast(state.d_positions.x.data()), 
                        thrust::raw_pointer_cast(state.d_positions.y.data()), 
                        thrust::raw_pointer_cast(state.d_positions.z.data()), 
                        thrust::raw_pointer_cast(sigma.data()), 
                        thrust::raw_pointer_cast(epsilon.data()), 
                        thrust::raw_pointer_cast(cutoff.data()), 
                        thrust::raw_pointer_cast(identifier.data()), 
                        state.n_atoms, 
                        num_species, 
                        cell
                    ), 
                    0.0f, 
                    thrust::plus<float>()
                );
            }

        private: 
            int num_species;
            thrust::device_vector<float> sigma;
            thrust::device_vector<float> epsilon;
            thrust::device_vector<float> cutoff;
            thrust::device_vector<int> identifier;
            
            CellType cell;
            md::utils::NeighbourList *NL;
    };
}

namespace md::utils::initialize {
    template <typename CellType>
    md::interactions::LJPotential<CellType> init_LJPotential_from_json(std::string json_path, State &state, CellType &cell, NeighbourList *NL) {
        using json = nlohmann::json;
        using string = std::string;

        std::ifstream f(json_path);
        if (!f.is_open()) {
            throw std::runtime_error("jsonファイルの読み込みに失敗しました。" );
        }

        json data = json::parse(f);

        if (!data.contains("lennard_jones")) {
            throw std::runtime_error("LJポテンシャルの設定が見つかりません。" );
        }

        auto lj_data = data["lennard_jones"];

        // データの読み込み
        std::vector<float> sigma = lj_data.at("sigma").get<std::vector<float>>();
        std::vector<float> epsilon = lj_data.at("epsilon").get<std::vector<float>>();
        std::vector<float> cutoff = lj_data.at("cutoff").get<std::vector<float>>();
        std::vector<int> numbers = lj_data.at("numbers").get<std::vector<int>>();

        int num_species = numbers.size();

        // GPUからデータ転送
        std::vector<int> atomic_numbers = state.d_atomic_numbers;

        // identifierの作成
        std::vector<int> identifier;
        identifier.reserve(atomic_numbers.size());

        int max_val = *std::max_element(numbers.begin(), numbers.end());
        std::vector<int> lut(max_val + 1, -1);

        for (int i = 0; i < num_species; ++i) {
            lut[numbers[i]] = i;
        }

        for (int num : atomic_numbers) {
            identifier.push_back(lut[num]);
        }

        return md::interactions::LJPotential(num_species, cell, NL, sigma, epsilon, cutoff, identifier);
    }
}

#endif