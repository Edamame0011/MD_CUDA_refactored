#ifndef NO_THERMOSTAT_CUH
#define NO_THERMOSTAT_CUH

#include <md/thermostats/Thermostat.cuh>
#include <md/core/State.cuh>

namespace md::thermostats {
    class NoThermostat : public Thermostat {
        void stepOne(State& state) override {}  // 何もしない
        void stepTwo(State& state) override {}  // 何もしない
    };
}

#endif