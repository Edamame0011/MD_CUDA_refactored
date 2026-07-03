#pragma once

#include <md/interactions/Interaction.cuh>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace md {
    class State;
    class Cell;
    class NeighbourList;
}

namespace md::interactions {
    class NNP_force_aoti : public Interaction {
        public: 
            NNP_force_aoti(State& state, Cell* _cell, NeighbourList* _nl, float _cutoff, int _num_max_edges, const std::string force_model_path, const std::string energy_model_path);
            ~NNP_force_aoti();

            void calc_force(State& state) override;
            void calc_potential(State& state) override;

        private: 
            void create_graph(State& state);

            const int num_max_edges;
            int num_edges;

            torch::inductor::AOTIModelPackageLoader force_loader;
            torch::inductor::AOTIModelPackageLoader energy_loader;

            NeighbourList* nl = nullptr;
            Cell* cell = nullptr;

            float cutoff;

            int* counts = nullptr;  // それぞれの原子のペア数 (N, )
            int* offsets = nullptr; // それぞれの原子の書き込み位置 (N, ) 

            // グラフ構造の本体
            int64_t* x_ptr = nullptr;
            float* edge_weight_ptr = nullptr;
            int64_t* edge_index_ptr = nullptr;

            // torch::Tensor型のラッパー
            std::vector<torch::Tensor> inputs;
    };
}