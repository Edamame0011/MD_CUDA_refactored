#pragma once

#include <md/convergence_checkers/ConvChecker.hpp>

namespace md::convergence_checkers {
    class MaxNorm : public ConvChecker {
        public:
            MaxNorm(float _threshold) : threshold(_threshold) {}
            bool check(State* state, SimState& simstate) override;
        private:
            float threshold = 0.0f;
    };
}