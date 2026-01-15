#include "core/Macro.h"
#include "shape/SizeComputer.hpp"

namespace MNN {
class MyUnarySizeComputer : public SizeComputer {
public:
  MyUnarySizeComputer() {
    MNN_PRINT("MyUnarySizeComputer()\n");
  }

  virtual bool
  onComputeSize(const Op *op, const std::vector<Tensor *> &inputs,
                const std::vector<Tensor *> &outputs) const override {
    MNN_PRINT("MyUnarySizeComputer onComputeSize()\n");
    MNN_ASSERT(1 == inputs.size());
    MNN_ASSERT(1 == outputs.size());
    // set output type & format
    auto input0 = inputs[0];
    auto output = outputs[0];
    auto &buffer = output->buffer();
    const auto opType = op->main_as_MyCustomUnaryOpParam()->funcType();

    buffer.type = input0->getType();

    TensorUtils::getDescribe(output)->dimensionFormat =
        TensorUtils::getDescribe(input0)->dimensionFormat;
    return SizeComputer::computeBroadCastDims(op, inputs, outputs);
  }
};
REGISTER_SHAPE(MyUnarySizeComputer, OpType_MyCustomUnary);
} // namespace MNN