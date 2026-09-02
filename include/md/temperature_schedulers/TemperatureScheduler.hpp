#pragma once

namespace md {
    class State;
    class SimState;

    class TemperatureScheduler {
        public:
            virtual ~TemperatureScheduler() = default;

            virtual void get_temperature(State& state, SimState& simstate) = 0;

        protected:
            TemperatureScheduler() = default;
    };
}
