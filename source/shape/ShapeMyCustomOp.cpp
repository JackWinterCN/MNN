#include "shape/SizeComputer.hpp"
#include "core/Macro.h"

namespace MNN {
class MyCustomOpSizeComputer : public SizeComputer {
public:
    virtual bool onComputeSize(const Op* op, const std::vector<Tensor*>& inputs,
                               const std::vector<Tensor*>& outputs) const override {
        MNN_ASSERT(2 == inputs.size());
        MNN_ASSERT(1 == outputs.size());
        // set output type & format
        auto input0 = inputs[0];
        auto input1 = inputs[1];
        auto output = outputs[0];
        auto &buffer = output->buffer();
        const auto opType = op->main_as_MyCustomOpParam()->opType();

        buffer.type = input0->getType();

        if (input0->getType().code != input1->getType().code) {
#ifdef DEBUG
            MNN_PRINT("Error for binary op: input0's type != input1's type, %d != %d, optype:%d, ", input0->getType().code, input1->getType().code, opType);
            if (nullptr != op->name()) {
                MNN_PRINT("op name: %s", op->name()->c_str());
            }
            MNN_PRINT("\n");
#endif
            return false;
        }

        if (input0->dimensions() < input1->dimensions()) {
            auto temp = input0;
            input0 = input1;
            input1 = temp;
        }
        TensorUtils::getDescribe(output)->dimensionFormat = TensorUtils::getDescribe(input0)->dimensionFormat;
        return SizeComputer::computeBroadCastDims(op, inputs, outputs);
    }
};
REGISTER_SHAPE(MyCustomOpSizeComputer, OpType_MyCustomOp);
} // namespace MNN