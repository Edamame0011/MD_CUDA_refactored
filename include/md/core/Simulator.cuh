#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/observers/Observer.cuh>
#include <md/utils/compute.cuh>

namespace md {
    template <typename CellType>
    class Simulator {
        public:
            // コンストラクタ
            Simulator(
                State& _state, 
                Interaction *_interaction, 
                Integrator *_integrator, 
                Observer *_observer, 
                CellType& _cell
            ) : state(_state), interaction(_interaction), integrator(_integrator), observer(_observer), cell(_cell) { }
        
            // シミュレーションの実行
            void run(float tsim) {
                /*
                // CUDA Graphsによる最適化のために必要な変数
                cudaGraph_t graph;
                cudaGraphExec_t instance;

                // 録画の開始
                // 複数回のループを一つのグラフとして記録する。
                constexpr int num_loop_per_graph = 100;
                cudaStreamBeginCapture(state.stream, cudaStreamCaptureModeGlobal);
                for (int i = 0; i < num_loop_per_graph; i ++) {
                    integrator->integrateStepOne(state);
                    cell.apply_pbc(state);
                    interaction->calc_force(state);
                    integrator->integrateStepTwo(state);
                }
                // 録画の終了
                cudaStreamEndCapture(state.stream, &graph);

                // グラフの変換
                cudaGraphInstantiate(&instance, graph, NULL, NULL, 0);
                */

                if (state.current_steps == 0) {
                    interaction->calc_force(state);
                    observer->init(state, this->interaction);
                }

                int total_steps = static_cast<int>(tsim / state.dt);
                total_steps += state.current_steps;

                // メインループ
                while (state.current_steps < total_steps) {  
                    integrator->integrateStepOne(state);
                    cell.apply_pbc(state);
                    interaction->calc_force(state);
                    integrator->integrateStepTwo(state);
                    state.current_steps ++;
                    // cudaGraphLaunch(instance, state.stream);
                    // state.current_steps += num_loop_per_graph;
                    observer->output(state, this->interaction);
                }

                // cudaGraphExecDestroy(instance);
                // cudaGraphDestroy(graph);
            };
        
        private:
            State& state;
            Interaction* interaction;
            Integrator* integrator;
            Observer* observer;
            CellType cell;
    };
}

#endif