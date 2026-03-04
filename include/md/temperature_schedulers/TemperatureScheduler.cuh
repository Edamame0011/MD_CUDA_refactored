#ifndef TEMPERATURE_SCHEDULER_CUH
#define TEMPERATURE_SCHEDULER_CUH

namespace md {
    class TemperatureScheduler {
        public: 
            virtual ~TemperatureScheduler() = default;
            
            virtual float get_temperature(const int step) = 0;

        protected:
            TemperatureScheduler() = default;
    };
}

#endif