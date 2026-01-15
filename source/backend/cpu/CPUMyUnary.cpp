#include "backend/cpu/CPUMyUnary.hpp"

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {

CPUMyUnary::CPUMyUnary(Backend *bn) : Execution(bn) {}

CPUMyUnary::CPUMyUnary(Backend *bn, int32_t op_type)
    : Execution(bn), myUnaryType(op_type) {}

CPUMyUnary::CPUMyUnary(Backend *bn, const MNN::Op *op,
                       const std::vector<Tensor *> &inputs,
                       const std::vector<Tensor *> &outputs)
    : Execution(bn),
      myUnaryType(op->main_as_MyCustomUnaryOpParam()->funcType()),
      mOperand(op->main_as_MyCustomUnaryOpParam()->operand()) {
  MNN_PRINT("[CPU] CPUMyUnary().\n");
}

ErrorCode CPUMyUnary::onResize(const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs) {
  mRealSize = inputs[0]->elementSize();
  MNN_PRINT("input ele number: %d\n", mRealSize);
  return NO_ERROR;
}

ErrorCode CPUMyUnary::onExecute(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) {
  auto input = inputs[0];
  auto output = outputs[0];

  auto input0Ptr = input->host<uint8_t>();
  auto outputPtr = output->host<uint8_t>();

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
      default: {
        break;
      }
    }
  }
  return NO_ERROR;
}

class CPUMyUnaryCreator : public CPUBackend::Creator {
public:
  virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const MNN::Op *op,
                              Backend *backend) const override {
    return new CPUMyUnary(backend, op, inputs, outputs);
  }
};

REGISTER_CPU_OP_CREATOR(CPUMyUnaryCreator, OpType_MyCustomUnary);

} // namespace MNN