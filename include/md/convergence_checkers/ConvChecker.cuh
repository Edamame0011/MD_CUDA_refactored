#pragma once

namespace md {
    class State;

    class ConvChecker {
        public:
            virtual ~ConvChecker() = default;
            virtual bool check(State& state) = 0;
        protected:
            ConvChecker() = default;
    };
}