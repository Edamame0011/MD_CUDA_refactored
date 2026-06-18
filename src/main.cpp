#include <md/utils/SimulationRunner.hpp>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    try {
        if (argc < 2) {
            std::cerr << "jsonファイルのパスを入力してください。" << std::endl;
            return 1;
        }
    
        std::string json_path = argv[1];

        md::utils::SimulationRunner runner(json_path);
        runner.run();
    
    } catch(const std::exception& e) {
        std::cerr << "[Error]" << e.what() << std::endl;
        return 1;
    } 
    return 0;
}