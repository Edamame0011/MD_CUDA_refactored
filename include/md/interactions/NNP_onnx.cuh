#ifndef __NNP_ONNX_CUH__
#define __NNP_ONNX_CUH__

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/utils/NeighbourList.cuh>
#include <torch/script.h>
#include <torch/torch.h>
#include <string>
#include <onnxruntime_cxx_api.h>

namespace md::interactions {
    template <typename CellType>
    class NNP_onnx : public Interaction {
        public: 
            NNP_onnx(State& state, CellType _cell, md::utils::NeighbourList<CellType>* _nl, float _cutoff, int _num_max_edges, const std::string model_path);
            ~NNP_onnx();

            void calc_force(State& state) override;
            void calc_potential(State& state) override;

        private: 
            void create_graph(State& state);

            const int num_max_edges;
            int num_edges;

            torch::jit::script::Module model;
            md::utils::NeighbourList<CellType>* nl;
            CellType cell;

            float cutoff;

            int* counts = nullptr;  // それぞれの原子のペア数 (N, )
            int* offsets = nullptr; // それぞれの原子の書き込み位置 (N, )

            // グラフ構造の本体
            int64_t* x_ptr = nullptr;
            float* edge_weight_ptr = nullptr;
            int64_t* edge_index_ptr = nullptr;

            float* total_energy_buffer = nullptr;
            float* forces_buffer = nullptr;

            // Ort::Value型のラッパー
            Ort::Value x;
            Ort::Value edge_weight;
            Ort::Value edge_index;

            Ort::Value total_energy;
            Ort::Value forces;

            // Ortの設定
            Ort::Env env;
            std::unique_ptr<Ort::Session> session;
            Ort::MemoryInfo cuda_memory_info;
            std::unique_ptr<Ort::IoBinding> io_binding;
    };
}

#endif