#pragma once

#include <md/thermostats/Thermostat.cuh>

namespace md {
    class State;
}

namespace md::thermostats {
    class NoThermostat : public Thermostat {
        public:
            void stepOne(State& state) override {}  // 何もしない
            void stepTwo(State& state) override {}  // 何もしない
    };
}