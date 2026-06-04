#pragma once

#include <md/observers/Observer.cuh>

namespace md::observers{
    class LogOutput : public Observer {
        public:
            LogOutput(float _interval, int _counter, Interaction* _interaction);
            void output(State& state) override;
            void init(State& state) override;
        private:
            float log_interval;
            int counter;
            float checker;
            Interaction* interaction;
    };
}