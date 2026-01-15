#include "XPUMyUnary.hpp"

#include <cmath>

namespace MNN {

XPUMyUnary::XPUMyUnary(Backend *bn) : Execution(bn) {
  // pass
}

XPUMyUnary::XPUMyUnary(Backend *bn, int32_t op_type)
    : Execution(bn), myUnaryType(op_type) {
  // pass
}

XPUMyUnary::XPUMyUnary(Backend *bn, const MNN::Op *op,
                       const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : Execution(bn),
      myUnaryType(op->main_as_MyCustomUnaryOpParam()->funcType()),
      mOperand(op->main_as_MyCustomUnaryOpParam()->operand()) {
  MNN_PRINT("[XPU] XPUMyUnary().\n");
}

ErrorCode XPUMyUnary::onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) {
  MNN_PRINT("XPUMyUnary onResize().\n");
  mRealSize = inputs[0]->elementSize();
  MNN_PRINT("input ele number: %d\n", mRealSize);
  return NO_ERROR;
}

ErrorCode XPUMyUnary::onExecute(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) {
  MNN_PRINT("XPUMyUnary onExecute().\n");
  auto input = inputs[0];
  auto output = outputs[0];

  auto input0Ptr = (uint8_t *)input->deviceId();
  auto outputPtr = (uint8_t *)output->deviceId();

  if (input0Ptr == nullptr || outputPtr == nullptr) {
    MNN_ERROR("null tensor mem, input0Ptr: %p, outputPtr: %p\n", input0Ptr,
              outputPtr);
    return NOT_SUPPORT;
  }

  int inpBytes = input->getType().bytes();
  int outBytes = output->getType().bytes();

  for (int i = 0; i < mRealSize; i++) {
    auto inp0 = input0Ptr + i * inpBytes;
    auto out = outputPtr + i * outBytes;
    switch (myUnaryType) {
      case MyCustomUnaryFuncType_M_UNARY_POW: {
        ((float *)out)[0] = powf(((float *)inp0)[0], mOperand);
        break;
      }
      case MyCustomUnaryFuncType_M_UNARY_SQUARE: {
        ((float *)out)[0] = ((float *)inp0)[0] * ((float *)inp0)[0];
        break;
      }
      default: {
        MNN_ERROR("XPUMyUnary not support op type: %d\n", myUnaryType);
        break;
      }
    }
  }
  return NO_ERROR;
}

REGISTER_XPU_OP_CREATOR(XPUMyUnary, OpType_MyCustomUnary);

} // namespace MNN