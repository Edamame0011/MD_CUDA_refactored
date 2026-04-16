#ifndef __FIRE_MINIMIZER_CUH__
#define __FIRE_MINIMIZER_CUH__

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/observers/Observer.cuh>
#include <md/energy_minimizers/EnergyMinimizer.cuh>
#include <md/convergence_checkers/ConvChecker.cuh>

namespace md::energy_minimizers {
    template <typename CellType>
    class FireMinimizer : public EnergyMinimizer {
        public:
            FireMinimizer(
                State& _state, 
                CellType& _cell, 
                Interaction* _interaction, 
                Observer* _observer, 
                ConvChecker* _checker
            );
            void set_hyper_parameters(
                int _n_max, 
                int _n_delay, 
                int _n_neg_max, 
                float _dt_start, 
                float _dt_max, 
                float _dt_min, 
                float _f_inc, 
                float _f_dec, 
                float _alpha_start, 
                float _f_alpha, 
                bool _initialdelay
            );
            void run() override;

        private:
            State& state;
            CellType& cell;
            Interaction* interaction = nullptr;
            Observer* observer = nullptr;
            ConvChecker* checker = nullptr;

            // ハイパーパラメータ
            int n_max = 1e+3;
            int n_delay = 1e+2;
            int n_neg_max;
            float dt_start = 5e-3f;
            float dt_max = 1e-3f;
            float dt_min = 5e-4f;
            float f_inc = 0e-3f;
            float f_dec = 0e-3f;
            float alpha_start;
            float f_alpha;
            bool initialdelay = false;
    };
}

#endif