#pragma once

namespace md {
    class EnergyMinimizer {
        public:
            virtual ~EnergyMinimizer() = default;
            virtual void run() = 0;
        protected:
            EnergyMinimizer() = default;
    };
}