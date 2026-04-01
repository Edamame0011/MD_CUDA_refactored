#ifndef OBSERVER_CUH
#define OBSERVER_CUH

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>

namespace md{
    class Observer {
        public:
            virtual ~Observer () = default;

            virtual void output(State& state, Interaction* interaction) = 0;
            virtual void init(State& state, Interaction* interaction) = 0;
        protected:
            Observer() = default;
    };
}

#endif

namespace md::observers {
    void print_energies(State& state, Interaction* interaction);
}