#ifndef __ENERGY_MINIMIZER_CUH__
#define __ENERGY_MINIMIZER_CUH__

namespace md {
    class EnergyMinimizer {
        public:
            virtual ~EnergyMinimizer() = default;
            virtual void run() = 0;
        protected:
            EnergyMinimizer() = default;
    };
}

#endif