#ifndef __OBSERVER_CUH__
#define __OBSERVER_CUH__

#include <md/core/State.cuh>
#include <md/interactions/Interaction.cuh>

namespace md{
    class Observer {
        public:
            virtual ~Observer () = default;

            virtual void output(State& state) = 0;
            virtual void init(State& state) = 0;
        protected:
            Observer() = default;
    };
}

namespace md::observers {
    void print_energies(State& state, Interaction* interaction);
}

#endif