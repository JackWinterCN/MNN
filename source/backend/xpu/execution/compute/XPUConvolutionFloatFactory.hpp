//
//  XPUConvolutionFloatFactory.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef XPU_CONVOLUTION_FLOAT_FACTORY_HPP
#define XPU_CONVOLUTION_FLOAT_FACTORY_HPP

#include <core/Backend.hpp>
namespace MNN {
namespace XPU {
class XPUConvolutionFloatFactory {
public:
    static Execution* create(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs, const MNN::Op* op,
                             Backend* backend);
};
} // namespace XPU
} // namespace MNN

#endif // XPU_CONVOLUTION_FLOAT_FACTORY_HPP
