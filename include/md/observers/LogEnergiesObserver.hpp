#pragma once

#include <md/observers/Observer.hpp>
#include <memory>
#include <string>

namespace md::observers{
    class EnergiesPrinter;

    class LogEnergiesObserver : public Observer {
        public:
            LogEnergiesObserver(float _interval, int _counter, Interaction* _interaction, const std::string& output_path);
            ~LogEnergiesObserver();
            void output(State& state, SimState& simstate) override;
            void init(State& state, SimState& simstate) override;
        private:
            float log_interval;
            int counter;
            float checker;
            std::unique_ptr<EnergiesPrinter> printer;
    };
}