#pragma once

namespace md {
    class State;
    class SimState;

    class ConvChecker {
        public:
            virtual ~ConvChecker() = default;
            virtual bool check(State* state, SimState& simstate) = 0;
        protected:
            ConvChecker() = default;
    };
}