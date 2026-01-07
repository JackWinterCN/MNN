#ifndef MNN_XPU_RUNNING_UTILS_HPP
#define MNN_XPU_RUNNING_UTILS_HPP

#include <vector>

#include <core/TensorUtils.hpp>

namespace MNN {
namespace XPU {

inline std::vector<int> tensorShapeFormat(const Tensor *input) {

  int iN =
      (0 != input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
  int iC =
      (0 != input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
  int iH =
      (0 != input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
  int iW =
      (0 != input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;

  if (input->buffer().dimensions >
      4) // more than 4 dimensions put to N dimension
  {
    for (int i = 4; i < input->buffer().dimensions; i++) {
      iW *= input->buffer().dim[i].extent;
    }
  }

  if (TensorUtils::getDescribe(input)->dimensionFormat ==
      MNN::MNN_DATA_FORMAT_NHWC) {
    iN =
        (0 < input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
    iH =
        (0 < input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
    iW =
        (0 < input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
    iC =
        (0 < input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;

    if (input->buffer().dimensions >
        4) // more than 4 dimensions put to N dimension
    {
      for (int i = 4; i < input->buffer().dimensions; i++) {
        iC *= input->buffer().dim[i].extent;
      }
    }
  }

  if (input->buffer().dimensions == 2) {
    iN = input->buffer().dim[0].extent;
    iH = 1;
    iW = 1;
    iC = input->buffer().dim[1].extent;
  }
  if (input->buffer().dimensions == 1) {
    iN = 1;
    iH = 1;
    iW = 1;
    iC = input->buffer().dim[0].extent;
  }

#ifdef LOG_VERBOSE
  MNN_PRINT("tensorShapeFormat : [%d, %d, %d, %d] \n", iN, iH, iW, iC);
#endif
  std::vector<int> shape_vec{iN, iH, iW, iC};

  return shape_vec;
}

inline DataType getDataType(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    if (nullptr == des->quantAttr.get()) {
        return DataType_DT_FLOAT;
    }
    return des->type;
}

} // namespace XPU
} // namespace MNN
#endif // MNN_XPU_BACKEND_HPP