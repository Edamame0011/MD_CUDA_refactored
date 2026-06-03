#include <md/observers/TargetTemperatureExporter.cuh>
#include <md/observers/TrajectoryExporter.cuh>
#include <md/core/constant.h>
#include <md/utils/compute.cuh>
#include <md/cells/CubicCell.cuh>
using namespace md::observers;

template <typename CellType>
void TargetTemperatureExporter<CellType>::output(State& state) {
    if (counter >= target_steps.size()) return;
    if (state.current_steps % 1000 == 0) {
        std::cout << "current step: " << state.current_steps << std::endl;
    }
    if (state.current_steps == target_steps[counter]) {
        // 温度の計算
        float K = md::utils::compute::calc_kinetic_energy(state);
        int dof = 3 * state.n_atoms;
        float temperature = 2 * K / (dof * boltzmann_constant);

        std::cout << "current temperature: " << temperature << std::endl;

        std::string output_path = output_folder_path + "output_" + std::to_string((int)target_temperatures[counter]) + ".xyz";
        TrajectoryExporter<CellType> exporter(state, output_path, cell);
        if (is_unwrap) {
            exporter.export_trajectory_unwrap(state);
        } else {
            exporter.export_trajectory(state);
        }
        counter ++;
    }
}

template class TargetTemperatureExporter<md::cells::CubicCell>;