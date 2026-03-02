#ifndef LOG_OUTPUT_CUH
#define LOG_OUTPUT_CUH

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>

namespace md::observers{
    class LogOutput : public Observer {
        public:
            LogOutput(float _interval, int _counter);
            void output(const State& state, const int step) override;
        private:
            float log_interval;
            int counter;
            float checker;
    };
}

#endif