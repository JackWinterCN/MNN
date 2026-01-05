#include "backend/cpu/CPUMyCustomOp.hpp"

#include "backend/cpu/CPUBackend.hpp"
#include "backend/cpu/compute/CommonOptFunction.h"

namespace MNN {

ErrorCode CPUMyCustomOp::onResize(const std::vector<Tensor *> &inputs,
                                  const std::vector<Tensor *> &outputs) {
    auto core = static_cast<CPUBackend*>(backend())->functions();
    mRealSize = static_cast<CPUBackend*>(backend())->getTensorSize(inputs[0]);
    mCacheDst.reset(mRealSize * core->bytes);
    mCacheSrc.reset(mRealSize * core->bytes);
    mOffset.reset(mRealSize * core->bytes);
    MNN_PRINT("CPUMyCustomOp::onResize() alloc: %d Bytes\n",
              mRealSize * core->bytes);
    return NO_ERROR;
}

ErrorCode CPUMyCustomOp::onExecute(const std::vector<Tensor *> &inputs,
                                   const std::vector<Tensor *> &outputs) {
  MNN_PRINT("CPUMyCustomOp::onExecute().\n");
  auto input = inputs[0];
  auto input1 = inputs[1];
  auto output = outputs[0];

  auto input0Ptr = input->host<uint8_t>();
  auto input1Ptr = input1->host<uint8_t>();
  auto outputPtr = output->host<uint8_t>();

  int inpBytes = input->getType().bytes();
  int outBytes = output->getType().bytes();
  if (halide_type_float == input->getType().code) {
    inpBytes = static_cast<CPUBackend *>(backend())->functions()->bytes;
  }
  if (halide_type_float == output->getType().code) {
    outBytes = static_cast<CPUBackend *>(backend())->functions()->bytes;
  }

  for (int i = 0; i < mRealSize; i++) {
    auto inp0 = input0Ptr + i * inpBytes;
    auto inp1 = input1Ptr + i * inpBytes;
    auto out = outputPtr + i * outBytes;
    switch (myCustomOpType) {
        case MyCustomOpType_M_ADD: {
          ((float *)out)[0] = ((float *)inp0)[0] + ((float *)inp1)[0];
          break;
        }
        default: {
          break;
        }
    }
  }
  return NO_ERROR;
}

class CPUMyCustomOpCreator : public CPUBackend::Creator {
public:
  virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const MNN::Op *op,
                              Backend *backend) const override {
    return new CPUMyCustomOp(backend, op->main_as_MyCustomOpParam()->opType());
  }
};

REGISTER_CPU_OP_CREATOR(CPUMyCustomOpCreator, OpType_MyCustomOp);

} // namespace MNN