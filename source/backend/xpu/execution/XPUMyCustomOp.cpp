#include "XPUMyCustomOp.hpp"

namespace MNN {

XPUMyCustomOp::XPUMyCustomOp(Backend *bn)
   : Execution(bn) {
}

XPUMyCustomOp::XPUMyCustomOp(Backend *bn, int32_t op_type)
    : Execution(bn), myCustomOpType(op_type) {
}

XPUMyCustomOp::XPUMyCustomOp(Backend *bn, const MNN::Op *op,
                             const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs)
    : Execution(bn), myCustomOpType(op->main_as_MyCustomOpParam()->opType()) {
  MNN_PRINT("[XPU] XPUMyCustomOp().\n");
}

ErrorCode XPUMyCustomOp::onResize(const std::vector<Tensor *> &inputs,
                                  const std::vector<Tensor *> &outputs) {
  MNN_PRINT("[XPU] XPUMyCustomOp::onResize().\n");

  // auto core = static_cast<CPUBackend*>(backend())->functions();
  mRealSize = inputs[0]->elementSize();
  // mCacheDst.reset(mRealSize * sizeof(float));
  // mCacheSrc.reset(mRealSize * sizeof(float));
  // mOffset.reset(mRealSize * sizeof(float));
  MNN_PRINT("[XPU] XPUMyCustomOp::onResize() input ele number: %d\n",
            mRealSize);
  return NO_ERROR;
}

ErrorCode XPUMyCustomOp::onExecute(const std::vector<Tensor *> &inputs,
                                   const std::vector<Tensor *> &outputs) {
  MNN_PRINT("[XPU] XPUMyCustomOp::onExecute().\n");
  auto input = inputs[0];
  auto input1 = inputs[1];
  auto output = outputs[0];

  auto input0Ptr = (uint8_t *)input->deviceId();
  auto input1Ptr = (uint8_t *)input1->deviceId();
  auto outputPtr = (uint8_t *)output->deviceId();
  // auto input0Ptr = input->host<uint8_t>();
  // auto input1Ptr = input1->host<uint8_t>();
  // auto outputPtr = output->host<uint8_t>();

  if (input0Ptr == nullptr || input1Ptr == nullptr || outputPtr == nullptr) {
    MNN_ERROR("null device mem, input0Ptr: %p, input1Ptr: %p, outputPtr: %p\n",
              input0Ptr, input1Ptr, outputPtr);
    return NOT_SUPPORT;
  }

  int inpBytes = input->getType().bytes();
  int outBytes = output->getType().bytes();

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

// class CPUMyCustomOpCreator : public CPUBackend::Creator {
// public:
//   virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
//                               const std::vector<Tensor *> &outputs,
//                               const MNN::Op *op,
//                               Backend *backend) const override {
//     return new XPUMyCustomOp(backend,
//     op->main_as_MyCustomOpParam()->opType());
//   }
// };

// REGISTER_CPU_OP_CREATOR(CPUMyCustomOpCreator, OpType_MyCustomOp);
REGISTER_XPU_OP_CREATOR(XPUMyCustomOp, OpType_MyCustomOp);

} // namespace MNN