// ======================= version 1 =======================
#include <MNN/ImageProcess.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExecutorScope.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/expr/Module.hpp>
#include <cv/cv.hpp>
#include <stdio.h>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;

int main(int argc, const char *argv[]) {
  if (argc < 3) {
    MNN_PRINT("Usage: %s float float\n", argv[0]);
    return 0;
  }

  VARP X = _Input({1, 1, 1}, NCHW);
  auto X_ptr = X->writeMap<float>();
  X_ptr[0] = atof(argv[1]);
  VARP Y = _Input({1, 1, 1}, NCHW);
  auto Y_ptr = Y->writeMap<float>();
  Y_ptr[0] = atof(argv[2]);
  // printf("X_ptr[0] = %f\n", X_ptr[0]);
  // printf("Y_ptr[0] = %f\n", Y_ptr[0]);

  MNN::Express::VARP Z = MNN::Express::_Add(X, Y);
  auto Z_ptr = Z->readMap<float>();
  printf("%f + %f = %f\n", X_ptr[0], Y_ptr[0], Z_ptr[0]);
}

// ======================= version 2 =======================
// #include <MNN/ImageProcess.hpp>
// #include <MNN/expr/Executor.hpp>
// #include <MNN/expr/ExprCreator.hpp>
// #include <MNN/expr/Module.hpp>
// #include <cv/cv.hpp>
// #include <stdio.h>

// using namespace MNN;
// using namespace MNN::Express;
// using namespace MNN::CV;

// int main(int argc, const char *argv[]) {
//   if (argc < 3) {
//     MNN_PRINT("Usage: ./yolov5_demo.out model.mnn input.jpg [forwardType] "
//               "[precision] [thread]\n");
//     return 0;
//   }

//   VARP inputs_0 = MNN::Express::_Input({1, 1}, NHWC, halide_type_of<float>());
//   VARP inputs_1 = MNN::Express::_Input({1, 1}, NHWC, halide_type_of<float>());

//   auto inputs_ptr_0 = inputs_0->writeMap<float>();
//   auto inputs_ptr_1 = inputs_1->writeMap<float>();

//   *inputs_ptr_0 = 5.5;
//   *inputs_ptr_1 = 3.4;

//   MNN::Express::VARP Y = MNN::Express::_Add(inputs_0, inputs_1);
//   const float *yPtr = Y->readMap<float>();
//   printf("Y = %f\n", *yPtr);

//   return 0;
// }