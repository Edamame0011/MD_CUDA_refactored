#ifndef LOG_OUTPUT_CUH
#define LOG_OUTPUT_CUH

#include <md/core/State.cuh>
#include <md/observers/Observer.cuh>

namespace md::observers{
    class LogOutput : public Observer {
        public:
            LogOutput(float _interval, int _counter);
            void output(State& state, Interaction* interaction) override;
            void init(State& state, Interaction* interaction) override;
        private:
            float log_interval;
            int counter;
            float checker;
    };
}

#endif