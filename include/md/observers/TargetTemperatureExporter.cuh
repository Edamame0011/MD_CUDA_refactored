#ifndef __TARGET_TEMPERATURE_EXPORTER_CUH__
#define __TARGET_TEMPERATURE_EXPORTER_CUH__

#include <md/observers/Observer.cuh>
#include <vector>

namespace md::observers {
    template <typename CellType>
    class TargetTemperatureExporter : public Observer {
        public:
            TargetTemperatureExporter(std::vector<float> _target_temperatures, float initial_temperature, float cooling_rate_per_step, std::string _output_folder_path, CellType _cell, bool _is_unwrap)
            : target_temperatures(_target_temperatures), output_folder_path(_output_folder_path),  cell(_cell), is_unwrap(_is_unwrap) {
                size_t size = _target_temperatures.size();
                target_steps.resize(size);

                for (size_t i = 0; i < size; i ++) {
                    float targ_tempr = target_temperatures[i];
                    size_t targ_step = (size_t)((initial_temperature - targ_tempr) / cooling_rate_per_step);
                    target_steps[i] = targ_step;
                    std::cout << "targ_step for " << targ_tempr << ": " << targ_step << std::endl;
                }
            }
            void output(State& state) override;
            void init(State& state) override { /*何もしない*/}
        private:
            std::vector<float> target_temperatures;
            std::vector<size_t> target_steps;
            std::string output_folder_path;
            CellType cell;
            bool is_unwrap;
            int counter = 0;
    };
}

#endif