#pragma once

namespace md {
    class State;
    class SimState;

    class Interaction {
        public:
            virtual ~Interaction () = default;

            virtual void calc_force(State& state, SimState& simstate) = 0;
            virtual float calc_potential(State& state, SimState& simstate) = 0;

        protected:
            Interaction() = default;
    };
}