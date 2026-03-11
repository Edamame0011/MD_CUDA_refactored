#ifndef SIMULATOR_CUH
#define SIMULATOR_CUH

#include <md/core/State.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/observers/Observer.cuh>

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
                CellType _cell
            ) : state(_state), interactions(_interaction), integrator(_integrator), observer(_observer), cell(_cell) { }
        
            // シミュレーションの実行
            void run(float tsim) {
                if (state.current_steps == 0) {
                    // 初期状態の処理
                    interactions->forward(state);
                    observer->init(state);
                }
                int total_steps = static_cast<int>(tsim / state.dt);
                total_steps += state.current_steps;
            
                // メインループ
                while (state.current_steps < total_steps) {
                    integrator->integrateStepOne(state);
                    cell.apply_pbc(state);

                    // 力のゼロ埋め
                    thrust::fill(state.d_forces.x.begin(), state.d_forces.x.end(), 0.0f);
                    thrust::fill(state.d_forces.y.begin(), state.d_forces.y.end(), 0.0f);
                    thrust::fill(state.d_forces.z.begin(), state.d_forces.z.end(), 0.0f);

                    interactions->forward(state);
                    integrator->integrateStepTwo(state);
                    
                    state.current_steps ++;
                
                    observer->output(state);
                }
            };
        
        private:
            State& state;
            Interaction* interactions;
            Integrator* integrator;
            Observer* observer;
            CellType cell;
    };
}

#endif