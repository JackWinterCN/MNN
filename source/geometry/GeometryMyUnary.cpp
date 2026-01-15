//
//  GeometryMyUnary.cpp
//  MNN
//
//  Created by MNN on 2020/07/27.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "core/OpCommonUtils.hpp"
#include "geometry/GeometryComputer.hpp"
#include "geometry/GeometryComputerUtils.hpp"

namespace MNN {
class GeometryMyUnary : public GeometryComputer {
public:
  GeometryMyUnary() { MNN_PRINT("GeometryMyUnary()\n"); }
  virtual bool onCompute(const Op *op, const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs, Context &context,
                         CommandBuffer &res) const override {
    MNN_PRINT("GeometryMyUnary onCompute()\n");
    // MNN_ASSERT(1 == inputs.size());
    // MNN_ASSERT(1 == outputs.size());
    // auto input = inputs[0];
    // auto output = outputs[0];
    // UnaryOpOperation unaryType;
    // switch (op->type()) {
    //     case OpType_TanH:
    //         unaryType = UnaryOpOperation_TANH;
    //         break;
    //     case OpType_Sigmoid:
    //         unaryType = UnaryOpOperation_SIGMOID;
    //         break;
    //     default:
    //         break;
    // }
    // auto cmd = GeometryComputerUtils::makeUnary(unaryType, input, output);
    // res.command.emplace_back(std::move(cmd));

    // auto &inputs = inputs;
    // Last Command
    std::shared_ptr<Command> cmdP(new Command);
    auto &cmd = *cmdP;
    cmd.op = op;
    cmd.inputs = std::move(inputs);
    cmd.outputs = std::move(outputs);
    res.command.emplace_back(std::move(cmdP));
    return true;
    return true;
  }
};

static void _create() {
  std::shared_ptr<GeometryComputer> comp(new GeometryMyUnary);
  GeometryComputer::registerGeometryComputer(comp, {OpType_MyCustomUnary});
}

REGISTER_GEOMETRY(GeometryMyUnary, _create);

} // namespace MNN
