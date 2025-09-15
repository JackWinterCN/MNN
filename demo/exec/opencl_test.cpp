#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/MNNForwardType.h>
#include <cv/cv.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    MNN_PRINT("Usage: %s model.mnn len\n", argv[0]);
    return 0;
  }

  const auto len_str = argv[2];
  const int len = std::stoi(len_str);

  int thread = 4;
  int precision = 0;
  int forwardType = MNN_FORWARD_OPENCL;

  MNN::ScheduleConfig sConfig;
  sConfig.type = static_cast<MNNForwardType>(forwardType);
  sConfig.numThread = thread;
  BackendConfig bConfig;
  bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
  sConfig.backendConfig = &bConfig;
  std::shared_ptr<Executor::RuntimeManager> rtmgr(
      Executor::RuntimeManager::createRuntimeManager(sConfig));
  if (rtmgr == nullptr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return 0;
  }
  // rtmgr->setCache(".cachefile");

  std::shared_ptr<Module> net(Module::load(
      std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));

  VARP X = _Input({1, 1, 1, len}, NCHW);
  auto X_ptr = X->writeMap<float>();
  VARP Y = _Input({1, 1, 1, len}, NCHW);
  auto Y_ptr = Y->writeMap<float>();
  for (int i = 0; i < len; i++) {
    X_ptr[i] = i * i;
    Y_ptr[i] = i + i;
  }
  X->getTensor()->print();
  Y->getTensor()->print();
  auto Z = net->onForward({X, Y});
  printf("outputs size: %ld\n", Z.size());
  auto Z_ptr = Z[0]->readMap<float>();
  Z[0]->getTensor()->print();
}