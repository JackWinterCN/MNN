#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(MyCustoUnaryOnnx);

MNN::OpType MyCustoUnaryOnnx::opType() {
  return MNN::OpType_MyCustomUnary;
}

MNN::OpParameter MyCustoUnaryOnnx::type() {
  return MNN::OpParameter_MyCustomUnaryOpParam;
}

void MyCustoUnaryOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                           OnnxScope *scope) {
  const auto &originalType = onnxNode->op_type();
  int inputSize = onnxNode->input_size();
  if (inputSize != 1) {
    DLOG(FATAL) << "Not support " << inputSize << " input for " << originalType
                << " op";
    return;
  }

  void *param = new MNN::MyCustomUnaryOpParamT;

  if (originalType == "M_UNARY_SQUARE") {
    ((MNN::MyCustomUnaryOpParamT *)param)->funcType = MNN::MyCustomUnaryFuncType_M_UNARY_SQUARE;
  } else if (originalType == "M_UNARY_POW") {
    ((MNN::MyCustomUnaryOpParamT *)param)->funcType = MNN::MyCustomUnaryFuncType_M_UNARY_POW;
  } else if (originalType == "M_UNARY_ABS") {
    ((MNN::MyCustomUnaryOpParamT *)param)->funcType = MNN::MyCustomUnaryFuncType_M_UNARY_ABS;
  } else {
    DLOG(FATAL) << "Not support " << originalType << " op";
    return;
  }

  if (onnxNode->attribute_size() > 0) {
    const auto &attributeProto = onnxNode->attribute(0);
    const auto &attributeName = attributeProto.name();
    if (attributeName == "operand") {
      DCHECK(attributeProto.type() == ::onnx::AttributeProto_AttributeType_INT)
          << "Node Attribute ERROR";
      ((MNN::MyCustomUnaryOpParamT *)param)->operand =
          static_cast<int>(attributeProto.i());
    }
  }

  dstOp->main.value = param;
}

REGISTER_CONVERTER(MyCustoUnaryOnnx, M_UNARY_SQUARE);
REGISTER_CONVERTER(MyCustoUnaryOnnx, M_UNARY_POW);
REGISTER_CONVERTER(MyCustoUnaryOnnx, M_UNARY_ABS);