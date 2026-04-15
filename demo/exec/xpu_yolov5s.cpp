#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/MNNDefine.h>
#include <cv/cv.hpp>
#include <perfetto_singleton.hpp>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

int main(int argc, const char *argv[]) {
  MNN_PRINT("run xpu_yolov5s.out\n");
  if (argc < 3) {
    MNN_PRINT("Usage: ./xpu_yolov5s.out model.mnn input.jpg [forwardType] "
              "[precision] [memory_mode] [thread]\n");
    return 0;
  }
  int thread = 4;
  int precision = BackendConfig::Precision_High;
  int memory_mode = BackendConfig::Memory_Normal;
  int forwardType = MNN_FORWARD_CPU;
  int warmup = 10;
  const auto model_file = argv[1];
  const auto input_file = argv[2];
  if (argc >= 4) {
    forwardType = atoi(argv[3]);
  }
  if (argc >= 5) {
    precision = atoi(argv[4]);
  }
  if (argc >= 6) {
    memory_mode = atoi(argv[5]);
  }
  if (argc >= 7) {
    thread = atoi(argv[6]);
  }

#ifdef MNN_PERFETTO_ENABLED
  // wait tracing start
  PerfettoSigleton::GetInstance().WaitForTracingStart();
#endif
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_BEGIN("MNN", "Init & Model Load");
#endif
  MNN::ScheduleConfig sConfig;
  sConfig.type = static_cast<MNNForwardType>(forwardType);
  sConfig.numThread = thread;
  BackendConfig bConfig;
  bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
  bConfig.memory = static_cast<BackendConfig::MemoryMode>(memory_mode);
  sConfig.backendConfig = &bConfig;
  std::shared_ptr<Executor::RuntimeManager> rtmgr =
      std::shared_ptr<Executor::RuntimeManager>(
          Executor::RuntimeManager::createRuntimeManager(sConfig));
  if (rtmgr == nullptr) {
    MNN_ERROR("Empty RuntimeManger\n");
    return 0;
  }
  rtmgr->setCache(".cachefile");

  std::shared_ptr<Module> net(Module::load(std::vector<std::string>{},
                                           std::vector<std::string>{},
                                           model_file, rtmgr));
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_END("MNN");
#endif
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_BEGIN("MNN", "Preprocess");
#endif
  auto original_image = imread(input_file);
  auto dims = original_image->getInfo()->dim;
  int ih = dims[0];
  int iw = dims[1];
  int len = ih > iw ? ih : iw;
  float scale = len / 640.0;
  std::vector<int> padvals{0, len - ih, 0, len - iw, 0, 0};
  auto pads = _Const(static_cast<void *>(padvals.data()), {3, 2}, NCHW,
                     halide_type_of<int>());
  auto image = _Pad(original_image, pads, CONSTANT);
  image = resize(image, Size(640, 640), 0, 0, INTER_LINEAR, -1, {0., 0., 0.},
                 {1. / 255., 1. / 255., 1. / 255.});
  auto input = _Unsqueeze(image, {0});
  input = _Convert(input, NC4HW4);
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_END("MNN");
#endif
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_BEGIN("MNN", "Warmup");
#endif
  for (int i = 0; i < warmup; i++) {
    MNN_PRINT("======> net forward warmup start\n");
    MNN::Timer _t;
    auto outputs = net->onForward({input});
    auto time = (float)_t.durationInUs() / 1000.0f;
    MNN_PRINT("======> net forward time = %f ms\n", time);
  }
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_END("MNN");
#endif
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_BEGIN("MNN", "Forward");
#endif
  MNN_PRINT("======> net forward start\n");
  MNN::Timer _t;
  auto outputs = net->onForward({input});
  auto time = (float)_t.durationInUs() / 1000.0f;
  MNN_PRINT("======> net forward time = %f ms\n", time);
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_END("MNN");
#endif
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_BEGIN("MNN", "Postprocess");
#endif
  auto output = _Convert(outputs[0], NCHW);
  output = _Squeeze(output);
  // output shape: [25200, 85]; 85 means: [cx, cy, w, h, box_conf, prob * 80]
  // get box_conf > 0.3 output
  MNN_PRINT("dims: \n");
  for (auto d : output->getInfo()->dim) {
    MNN_PRINT("%d\n", d);
  }
  auto box_conf = _GatherV2(output, _Scalar<int>(4), _Scalar<int>(1));
  auto has_object = _Greater(box_conf, _Scalar<float>(0.1));
  auto idx = Express::_Where(has_object);
  // idx->getTensor()->print();
  output = Express::_GatherND(output, idx);
  // idx->getTensor()->print();
  MNN_PRINT("dims: \n");
  for (auto d : output->getInfo()->dim) {
    MNN_PRINT("%d\n", d);
  }
  output = _Transpose(output, {1, 0}); // to [85, 25200]
  auto cx = _Gather(output, _Scalar<int>(0));
  auto cy = _Gather(output, _Scalar<int>(1));
  auto w = _Gather(output, _Scalar<int>(2));
  auto h = _Gather(output, _Scalar<int>(3));
  auto _box_conf = _Gather(output, _Scalar<int>(4));
  std::vector<int> startvals{5, 0};
  auto start = _Const(static_cast<void *>(startvals.data()), {2}, NCHW,
                      halide_type_of<int>());
  std::vector<int> sizevals{-1, -1};
  auto size = _Const(static_cast<void *>(sizevals.data()), {2}, NCHW,
                     halide_type_of<int>());
  auto probs = _Slice(output, start, size);
  // [cx, cy, w, h] -> [y0, x0, y1, x1]
  auto x0 = cx - w * _Const(0.5);
  auto y0 = cy - h * _Const(0.5);
  auto x1 = cx + w * _Const(0.5);
  auto y1 = cy + h * _Const(0.5);
  auto boxes = _Stack({x0, y0, x1, y1}, 1);
  auto scores = _ReduceMax(probs, {0});
  auto ids = _ArgMax(probs, 0);
  auto result_ids = _Nms(boxes, scores, 100, 0.45, 0.25);
  auto result_ptr = result_ids->readMap<int>();
  auto box_ptr = boxes->readMap<float>();
  auto box_conf_ptr = box_conf->readMap<float>();
  auto ids_ptr = ids->readMap<int>();
  auto score_ptr = scores->readMap<float>();
  for (int i = 0; i < 100; i++) {
    auto idx = result_ptr[i];
    if (idx < 0)
      break;
    auto x0 = box_ptr[idx * 4 + 0] * scale;
    auto y0 = box_ptr[idx * 4 + 1] * scale;
    auto x1 = box_ptr[idx * 4 + 2] * scale;
    auto y1 = box_ptr[idx * 4 + 3] * scale;
    auto class_idx = ids_ptr[idx];
    auto score = score_ptr[idx];
    MNN_PRINT("### box: {%f, %f, %f, %f}, class_idx: %d, score: %f\n", x0, y0, x1,
           y1, class_idx, score);
    rectangle(original_image, {x0, y0}, {x1, y1}, {0, 0, 255}, 2);
  }
  if (imwrite("res.jpg", original_image)) {
    MNN_PRINT("result image write to `res.jpg`.\n");
  }
  rtmgr->updateCache();
#ifdef MNN_PERFETTO_ENABLED
  TRACE_EVENT_END("MNN");
#endif
  return 0;
}