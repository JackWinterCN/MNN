#include <math.h>
#include <fstream>
#include <iostream>
#include <memory>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>

#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>

using namespace MNN;
using namespace MNN::Express;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "Usage: ./xpu_unary_test.out model.mnn len forwardType"
              << std::endl;
    return -1;
  }

  const auto modelFile = argv[1];
  const auto len_str = argv[2];
  const int len1 = std::stoi(len_str);
  int thread = 4;
  int precision = BackendConfig::Precision_High;
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
  if (rtmgr == nullptr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return 0;
  }
  std::shared_ptr<Module> net(Module::load(std::vector<std::string>{},
                                           std::vector<std::string>{},
                                           modelFile, rtmgr));
  VARP X = _Input({1, 1, 1, len1}, NCHW);
  auto X_ptr = X->writeMap<float>();
  for (int i = 0; i < len1; i++) {
    X_ptr[i] = i + 1;
    printf("X_ptr[%d] = %f\t", i, X_ptr[i]);
  }
  printf("\n");

  auto Z = net->onForward({X});
  if (Z.empty()) {
    MNN_ERROR("Z is empty\n");
    return 0;
  }
  MNN_PRINT("Z[0]->getInfo()->size = %ld\n", Z[0]->getInfo()->size);
  auto Z_ptr = Z[0]->readMap<float>();
  for (int i = 0; i < Z[0]->getInfo()->size; i++) {
    printf("Z_ptr[%d] = %f\t", i, Z_ptr[i]);
  }
  printf("\n");

  return 0;
}
