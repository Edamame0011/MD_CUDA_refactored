#pragma once

#include <md/observers/Observer.hpp>
#include <memory>

namespace md{    
    class Interaction;
    
    namespace observers{
        class EnergiesPrinter;
    
        class LinearEnergiesObserver : public Observer {
            public:
                LinearEnergiesObserver(int interval, Interaction* _interaction, const std::string& output_path);
                ~LinearEnergiesObserver();
                void output(State& state, SimState& simstate) override;
                void init(State& state, SimState& simstate) override;
            private:
                int output_interval;
                std::unique_ptr<EnergiesPrinter> printer;
        };
    }
}