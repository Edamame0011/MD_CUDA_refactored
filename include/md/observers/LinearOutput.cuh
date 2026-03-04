#ifndef LINEAR_OUTPUT_CUH
#define LINEAR_OUTPUT_CUH

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>

namespace md::observers{
    class LinearOutput : public Observer {
        public:
            LinearOutput(int interval) : output_interval(interval) {}
            void output(const State& state) override;
        private:
            int output_interval;
    };
}

#endif