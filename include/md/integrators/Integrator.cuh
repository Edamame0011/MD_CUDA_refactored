#pragma once

namespace md {
    class State;

    
    class Integrator {
        public:
            virtual ~Integrator() = default;
    
            virtual void integrateStepOne(State& state) = 0;    // 1段目の更新
            virtual void integrateStepTwo(State& state) = 0;    // 2段目の更新

        protected:
            Integrator() = default;
    };
}