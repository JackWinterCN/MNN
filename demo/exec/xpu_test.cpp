#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <MNN/Interpreter.hpp>
#include <MNN/AutoTime.hpp>

#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Executor.hpp>

using namespace MNN;
using namespace MNN::Express;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: ./xpu_test.out model.mnn len forwardType" << std::endl;
        return -1;
    }

    const auto modelFile = argv[1];
    const auto len_str  = argv[2];
    const int len = std::stoi(len_str);
    int thread = 4;
    int precision = 0;
    int forwardType = MNN_FORWARD_CPU;
    if (argc > 3) {
        forwardType = std::stoi(argv[3]);
    }

    MNN::ScheduleConfig sConfig;
    sConfig.type = static_cast<MNNForwardType>(forwardType);
    sConfig.numThread = thread;
    BackendConfig bConfig;
    bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
    sConfig.backendConfig = &bConfig;

    std::shared_ptr<Executor::RuntimeManager> rtmgr =
        std::shared_ptr<Executor::RuntimeManager>(
            Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }
    std::shared_ptr<Module> net(Module::load(std::vector<std::string>{},
                                             std::vector<std::string>{},
                                             modelFile, rtmgr));
    VARP X = _Input({1, 1, 1, len}, NCHW);
    auto X_ptr = X->writeMap<float>();
    VARP Y = _Input({1, 1, 1, len}, NCHW);
    auto Y_ptr = Y->writeMap<float>();
    for (int i = 0; i < len; i++) {
        X_ptr[i] = i*i;
        Y_ptr[i] = i+i;
    }
    for (int i = 0; i < len; i++) {
      printf("X_ptr[%d] = %f\t", i, X_ptr[i]);
    }
    printf("\n");
    for (int i = 0; i < len; i++) {
      printf("Y_ptr[%d] = %f\t", i, Y_ptr[i]);
    }
    printf("\n");

    auto Z = net->onForward({X, Y});

    auto Z_ptr = Z[0]->readMap<float>();
    for (int i = 0; i < len; i++) {
        printf("Z_ptr[%d] = %f\t", i, Z_ptr[i]);
    }
    printf("\n");

    return 0;
}
