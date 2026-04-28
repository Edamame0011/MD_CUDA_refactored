#ifndef INITIALIZE_FROM_JSON_CUH
#define INITIALIZE_FROM_JSON_CUH

#include <md/core/State.cuh>
#include <string>
#include <random>
#include <array>
#include <random>
#include <external/nlohmann/json.hpp>

#include <md/core/constant.h>
#include <md/core/State.cuh>
#include <md/core/Simulator.cuh>
#include <md/integrators/ConstantVolume.cuh>
#include <md/interactions/LJPotential.cuh>
#include <md/interactions/LJPotential_CLL.cuh>
#include <md/integrators/LangevinIntegrator.cuh>
#include <md/observers/LinearOutput.cuh>
#include <md/thermostats/NoThermostat.cuh>
#include <md/thermostats/NHC1.cuh>
#include <md/thermostats/BussiThermostat.cuh>
#include <md/cells/CubicCell.cuh>
#include <md/utils/NeighbourList.cuh>
#include <md/utils/initialize.cuh>
#include <md/observers/LogOutput.cuh>
#include <md/temperature_schedulers/TemperatureScheduler.cuh>
#include <md/temperature_schedulers/ConstantScheduler.cuh>
#include <md/temperature_schedulers/LinearScheduler.cuh>

using json = nlohmann::json;

namespace md::utils::initialize {
    void configure_units(const json& m_setting) {
        std::string unit_type = m_setting.value("unit", "lj");
        if (unit_type == "lj") {
            md::conversion_factor = 1.0;
            md::boltzmann_constant = 1.0;
        } else if (unit_type == "metal") {
            md::boltzmann_constant = 8.617333262145e-5f;
            md::conversion_factor = 0.964855e-2f;
        } else {
            throw std::runtime_error("未対応のunitです: " + unit_type);
        }
    }

    std::unique_ptr<md::State> init_state(std::array<std::array<float, 3>, 3>& lattice, const json& a_setting, std::mt19937& mt) {
        std::string mode = a_setting.value("mode", "");
        std::unique_ptr<md::State> state;
        
        if (mode == "generate_binary_lj") {
            int n_atoms = a_setting.at("n_atoms").get<int>();
            float density = a_setting.at("density").get<float>();
            auto ratio_vec = a_setting.at("ratio").get<std::vector<float>>();
            if (ratio_vec.size() < 2) throw std::runtime_error("ratioには少なくとも2つの要素が必要です。");
            
            float a_ratio = ratio_vec[0] / (ratio_vec[0] + ratio_vec[1]);
            state = md::utils::initialize::generate_binary_lj(n_atoms, density, lattice, a_ratio, mt);
        /*
        } else if (mode == "from_file") {
            std::string format = a_setting.value("format", "xyz");
            if (format == "xyz") {
                state = md::utils::initialize::read_state_from_xyz(lattice, a_setting.at("path"));
            } else {
                throw std::runtime_error("未対応のファイルフォーマットです: " + format);
            }
        */
        } else {
            throw std::runtime_error("未対応のatoms modeです: " + mode);
        }
        return state;
    }

    std::unique_ptr<md::Observer> build_observer(const json& o_setting) {
        std::string o_type = o_setting.value("type", "linear");
        if (o_type == "linear") {
            return std::make_unique<md::observers::LinearOutput>(o_setting.at("interval"));
        } else if (o_type == "log") {
            int divisions = o_setting.at("divisions");
            float log_interval = std::pow(10.0f, 1.0f / (float)divisions);
            return std::make_unique<md::observers::LogOutput>(log_interval, 5);
        }
        throw std::runtime_error("未対応のoutput typeです: " + o_type);
    }
    struct EnsembleComponents {
        std::unique_ptr<md::TemperatureScheduler> scheduler;
        std::unique_ptr<md::Thermostat> thermostat;
        std::unique_ptr<md::Integrator> integrator;
    }; 
    EnsembleComponents build_ensemble(const json& e_setting, md::State& state, std::mt19937& mt) {
        EnsembleComponents comp;
        std::string ensemble = e_setting.value("type", "NVE");
        
        if (ensemble == "NVE") {
            md::utils::initialize::init_velocities(state, e_setting.at("temperature"), mt);
            comp.thermostat = std::make_unique<md::thermostats::NoThermostat>();
            comp.integrator = std::make_unique<md::integrators::ConstantVolume>(comp.thermostat.get());
        } else if (ensemble == "NVT") {
            md::utils::initialize::init_velocities(state, e_setting.at("temperature"), mt);
            
            // Schedulerの構築
            std::string sched_type = e_setting.value("scheduler", "constant");
            if (sched_type == "constant") {
                comp.scheduler = std::make_unique<md::temperature_schedulers::ConstantScheduler>(e_setting.at("temperature"));
            } else if (sched_type == "linear") {
                float rate_per_step = (float)e_setting.at("rate_per_unit_time") * state.dt;
                comp.scheduler = std::make_unique<md::temperature_schedulers::LinearScheduler>(e_setting.at("temperature"), rate_per_step);
            } else {
                throw std::runtime_error("未対応のschedulerです: " + sched_type);
            }

            // Thermostatの構築
            std::string thermo_type = e_setting.value("thermostat", "Nose-Hoover");
            if (thermo_type == "Nose-Hoover") {
                auto nhc = std::make_unique<md::thermostats::NHC1>(
                    e_setting.value("tau", 1.0f), comp.scheduler.get()
                );
                nhc->init(state);
                comp.thermostat = std::move(nhc);
                comp.integrator = std::make_unique<md::integrators::ConstantVolume>(comp.thermostat.get());
            } 
            else if (thermo_type == "Bussi") {
                float tau = e_setting.value("tau", 1.0f); 
                int seed = e_setting.value("seed", 12345);
                auto bussi = std::make_unique<md::thermostats::BussiThermostat>(tau, comp.scheduler.get());
                bussi->init(state, seed);
                comp.thermostat = std::move(bussi);
                comp.integrator = std::make_unique<md::integrators::ConstantVolume>(comp.thermostat.get());
            } 
            else if (thermo_type == "Langevin") {
                float gamma = 1.0f / e_setting.value("tau", 1.0f);
                int seed = e_setting.value("seed", 12345);
                auto langevin = std::make_unique<md::integrators::LangevinIntegrator>(gamma, seed, comp.scheduler.get());
                langevin->init(state, seed);
                comp.integrator = std::move(langevin);
            }
            else {
                throw std::runtime_error("未対応のthermostatです: " + thermo_type);
            }
            
            
        } else {
            throw std::runtime_error("未対応のensembleです: " + ensemble);
        }
        return comp;
    }
    template <typename CellType>
    std::unique_ptr<md::Interaction> build_interaction(const json& p_setting, md::State& state, CellType& cell, md::utils::NeighbourList<CellType>* nl) {
        std::string p_type = p_setting.at("type");
        if (p_type == "lennard_jones") {
            auto pot = init_LJPotential_from_json(p_setting, state, cell, nl);
            return pot;
        /*
        } else if (p_type == "NNP_TorchScript") {
            return std::make_unique<md::interactions::NNPTorchScript<CellType>>(state, cell, p_setting.at("cutoff"), nl, p_setting.at("model_path"));
        }
        */
        } else throw std::runtime_error("未対応のpotential typeです: " + p_type);
    }

    std::unique_ptr<md::Interaction> build_interaction(const json& p_setting, md::State& state, md::cells::CubicCell& cell, md::utils::NeighbourList_CLL* nl) {
        std::string p_type = p_setting.at("type");
        if (p_type == "lennard_jones") {
            auto pot = init_LJPotential_CLL_from_json(p_setting, state, cell, nl);
            return pot;
        /*
        } else if (p_type == "NNP_TorchScript") {
            return std::make_unique<md::interactions::NNPTorchScript<CellType>>(state, cell, p_setting.at("cutoff"), nl, p_setting.at("model_path"));
        }
        */
        } else throw std::runtime_error("未対応のpotential typeです: " + p_type);
    }
}

#endif