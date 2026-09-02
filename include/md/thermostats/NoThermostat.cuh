#pragma once

#include <md/thermostats/Thermostat.cuh>

namespace md {
    class State;

    namespace thermostats {
        class NoThermostat : public Thermostat {
            public:
                void stepOne(State& state, SimState& simstate) override {}  // 何もしない
                void stepTwo(State& state, SimState& simstate) override {}  // 何もしない
        };
    }
}