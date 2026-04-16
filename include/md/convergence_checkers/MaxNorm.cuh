#ifndef __MAX_NORM_CUH__
#define __MAX_NORM_CUH__

#include <md/convergence_checkers/ConvChecker.cuh>

namespace md::convergence_checkers {
    class MaxNorm : public ConvChecker {
        public:
            MaxNorm(float _threshold) : threshold(_threshold) {}
            bool check(State& state) override;
        private:
            float threshold = 0.0f;
    };
}

#endif