#pragma once

#include <md/energy_minimizers/EnergyMinimizer.hpp>

namespace md {
    class State;
    class SimState;
    class Interaction;
    class Observer;
    class Cell;
    class ConvChecker;

    namespace energy_minimizers {
        class FireMinimizer : public EnergyMinimizer {
            public:
                FireMinimizer(
                    State* state, 
                    SimState& simstate, 
                    Cell* cell, 
                    Interaction* interaction, 
                    Observer* observer, 
                    ConvChecker* checker
                );
                void set_hyper_parameters(
                    int _n_max, 
                    int _n_delay, 
                    int _n_neg_max, 
                    float _dt_start, 
                    float _t_max, 
                    float _t_min, 
                    float _f_inc, 
                    float _f_dec, 
                    float _alpha_start, 
                    float _f_alpha, 
                    bool _initialdelay
                );
                void run() override;

            private:
                State* state_;
                SimState& simstate_;
                Cell* cell_ = nullptr;
                Interaction* interaction_ = nullptr;
                Observer* observer_ = nullptr;
                ConvChecker* checker_ = nullptr;

                // ハイパーパラメータ
                int n_max = 5e+3;
                int n_delay = 20;
                int n_neg_max = 2000;
                float dt_start = 5e-3f;
                float t_max = 5.0f;
                float t_min = 0.02f;
                float f_inc = 1.1f;
                float f_dec = 0.5f;
                float alpha_start = 0.25f;
                float f_alpha = 0.99f;
                bool initialdelay = true;
        };
    }
}