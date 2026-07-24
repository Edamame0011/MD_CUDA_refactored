#pragma once

#include <string>
#include <random>
#include <memory>
#include <filesystem>

#include <external/nlohmann/json.hpp>

namespace md {
    class State;
    class Integrator;
    class Interaction;
    class Observer;
    class Cell;
    class NeighbourList;
    class TemperatureScheduler;
    class Thermostat;
    class ConvChecker;
    class EnergyMinimizer;
    class SortedCellList;
    class NeighbourList_CLL;
    class UnsortedCellList;
    class NeighbourList_uCLL;
}

namespace md::utils {
    class SimulationRunner {
        public:
            SimulationRunner(const std::string& setting_path);
            ~SimulationRunner();

            void run();

        private:
            nlohmann::json j;

            std::unique_ptr<State> state;
            std::unique_ptr<Integrator> integrator;
            std::unique_ptr<Interaction> interaction;
            std::unique_ptr<Observer> observer;
            std::unique_ptr<Cell> cell;
            std::unique_ptr<NeighbourList> nl;

            std::unique_ptr<TemperatureScheduler> scheduler;
            std::unique_ptr<Thermostat> thermostat;

            std::unique_ptr<ConvChecker> checker;
            std::unique_ptr<EnergyMinimizer> minimizer;

            std::unique_ptr<SortedCellList> cll;
            std::unique_ptr<NeighbourList_CLL> nl_cll;

            std::unique_ptr<UnsortedCellList> ucll;
            std::unique_ptr<NeighbourList_uCLL> nl_ucll;

            std::array<std::array<float, 3>, 3> lattice;
            std::mt19937 mt;

            // すべての親ディレクトリ
            std::filesystem::path parent_dir;
            // ステップ毎のディレクトリ
            std::filesystem::path step_dir;

            void configure_units(const nlohmann::json& m_setting);
            void build_state(const nlohmann::json& a_setting);
            void build_observer(const nlohmann::json& o_setting);
            void build_ensemble(const nlohmann::json& e_setting);
            void build_interaction(const nlohmann::json& i_setting);
            void build_checker(const nlohmann::json& ch_setting);
            void build_minimizer(const nlohmann::json & mi_setting);
    };
}