#pragma once

namespace md{
    class State;
    class SimState;

    class Observer {
        public:
            virtual ~Observer () = default;

            virtual void output(State& state, SimState& simstate) = 0;
            virtual void init(State& state, SimState& simstate) = 0;
        protected:
            Observer() = default;
    };
}