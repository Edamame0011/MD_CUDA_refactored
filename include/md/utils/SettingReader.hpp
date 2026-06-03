#ifndef __SETTING_PARSER_CUH__
#define __SETTING_PARSER_CUH__

#include <string>
#include <external/nlohmann/json.hpp>

#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/core/Simulator.cuh>
#include <md/integrators/Integrator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/integrators/LangevinIntegrator.cuh>
#include <md/interactions/LJPotential.cuh>
#include <md/interactions/LJPotential_CLL.cuh>
#include <md/interactions/NNP.cuh>
#include <md/interactions/NNP_CSR.cuh>
#include <md/interactions/NNP_aoti.cuh>
#include <md/observers/LinearOutput.cuh>
#include <md/observers/LogOutput.cuh>
#include <md/observers/LinearExportTrajectory.cuh>
#include <md/observers/LogExportTrajectory.cuh>
#include <md/observers/TargetTemperatureExporter.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/thermostats/NHC1.cuh>
#include <md/thermostats/BussiThermostat.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/initialize.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/temperature_schedulers/ConstantScheduler.cuh>
#include <md/temperature_schedulers/LinearScheduler.cuh>

namespace md::utils {
    class SettingReader {
        public:
            SettingReader(std::string setting_path);

            State* state_ptr() { return state.get(); }
            Integrator* integrator_ptr() { return integrator.get(); }
            Interaction* interaction_ptr() { return interaction.get(); }
            Observer* ovserver_ptr() { return observer.get(); }

        private:
            std::unique_ptr<State> state = nullptr;
            std::unique_ptr<Integrator> integrator = nullptr;
            std::unique_ptr<Interaction> interaction = nullptr;
            std::unique_ptr<Observer> observer = nullptr;

            std::unique_ptr<md::TemperatureScheduler> scheduler = nullptr;
            std::unique_ptr<md::Thermostat> thermostat = nullptr;
    };
}

#endif