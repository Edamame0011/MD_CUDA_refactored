#ifndef LJ_POTENTIAL_CUH
#define LJ_POTENTIAL_CUH

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/utils/NeighbourList.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>
#include <md/cells/Cell.cuh>

#include <external/nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <string>

struct lj_params {
    float *sigma, *epsilon, *cutoff, *deriv_1st_LJpotential_cutoff;
    int *identifier;
};

namespace md::interactions {
    class LJPotential : public Interaction {
        public: 
            LJPotential(
                int _num_species, 
                Cell* _cell, 
                NeighbourList *_NL, 
                std::vector<float> _sigma, 
                std::vector<float> _epsilon, 
                std::vector<float> _cutoff, 
                std::vector<int> _identifier
            );
            ~LJPotential();

            void calc_force(State& state) override;
            void calc_potential(State& state) override;

        private: 
            int num_species;
            lj_params params;
            
            Cell* cell;
            NeighbourList *nl;

            // カーネル起動スレッド数
            int calc_force_num_threads;
    };
}

namespace md::utils::initialize {
    template <typename CellType>
    std::unique_ptr<md::interactions::LJPotential> init_LJPotential_from_json(const nlohmann::json& json, State &state, Cell* cell, NeighbourList* NL) {
        // データの読み込み
        std::vector<float> sigma = json.at("sigma").get<std::vector<float>>();
        std::vector<float> epsilon = json.at("epsilon").get<std::vector<float>>();
        std::vector<float> cutoff = json.at("cutoff").get<std::vector<float>>();
        std::vector<int> numbers = json.at("numbers").get<std::vector<int>>();

        int num_species = numbers.size();

        // GPUからデータ転送
        int N = state.n_atoms;
        std::vector<int> atomic_numbers(N);
        cudaMemcpy(atomic_numbers.data(), state.atomic_numbers, N * sizeof(int), cudaMemcpyDeviceToHost);

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

        return std::make_unique<md::interactions::LJPotential>(num_species, cell, NL, sigma, epsilon, cutoff, identifier);
    }
}

#endif