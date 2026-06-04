#pragma once

namespace md {
    class State;

    
    class Interaction {
        public:
            virtual ~Interaction () = default;

            virtual void calc_force(State& state) = 0;
            virtual void calc_potential(State& state) = 0;

        protected:
            Interaction() = default;
    };
}