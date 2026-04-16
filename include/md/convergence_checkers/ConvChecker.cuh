#ifndef __CONV_CHECKER_CUH__
#define __CONV_CHECKER_CUH__

#include <md/core/State.cuh>

namespace md {
    class ConvChecker {
        public:
            virtual ~ConvChecker() = default;
            virtual bool check(const State& state) = 0;
        protected:
            ConvChecker() = default;
    };
}

#endif