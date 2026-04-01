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
                CellType _cell
            ) : state(_state), interaction(_interaction), integrator(_integrator), observer(_observer), cell(_cell) { }
        
            // シミュレーションの実行
            void run(float tsim) {
                auto view = state.get_view();
                auto N = state.n_atoms;

                if (state.current_steps == 0) {
                    thrust::fill(thrust::device, view.force.x, view.force.x + N, 0.0f);
                    thrust::fill(thrust::device, view.force.y, view.force.y + N, 0.0f);
                    thrust::fill(thrust::device, view.force.z, view.force.z + N, 0.0f);

                    interaction->calc_force(state);
                    observer->init(state, this->interaction);
                }
                int total_steps = static_cast<int>(tsim / state.dt);
                total_steps += state.current_steps;

                // メインループ
                while (state.current_steps < total_steps) {  
                    integrator->integrateStepOne(state);

                    cell.apply_pbc(state);

                    thrust::fill(thrust::device, view.force.x, view.force.x + N, 0.0f);
                    thrust::fill(thrust::device, view.force.y, view.force.y + N, 0.0f);
                    thrust::fill(thrust::device, view.force.z, view.force.z + N, 0.0f);

                    interaction->calc_force(state);

                    integrator->integrateStepTwo(state);

                    state.current_steps ++;
                
                    observer->output(state, this->interaction);

                    md::utils::compute::remove_drift(state);
                }
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