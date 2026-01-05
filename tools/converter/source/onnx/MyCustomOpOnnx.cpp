#include "onnxOpConverter.hpp"

DECLARE_OP_CONVERTER(MyCustomOpOnnx);

MNN::OpType MyCustomOpOnnx::opType() {
    return MNN::OpType_MyCustomOp;
}

MNN::OpParameter MyCustomOpOnnx::type() {
  return MNN::OpParameter_MyCustomOpParam;
}

void MyCustomOpOnnx::run(MNN::OpT *dstOp, const onnx::NodeProto *onnxNode,
                         OnnxScope *scope) {
  const auto &originalType = onnxNode->op_type();
  int inputSize = onnxNode->input_size();
  if (inputSize != 2) {
    DLOG(FATAL) << "Not support 1 input for " << originalType << " op";
    return;
  }

  void *param = new MNN::MyCustomOpParamT;

#define TO_BINARY_OP(src, dst)                                                 \
  if (originalType == src) {                                                   \
    ((MNN::MyCustomOpParamT *)param)->opType = dst;                            \
  }

  TO_BINARY_OP("M_ADD", MNN::MyCustomOpType_M_ADD);
  TO_BINARY_OP("M_SUB", MNN::MyCustomOpType_M_SUB);
  TO_BINARY_OP("M_MUL", MNN::MyCustomOpType_M_MUL);
  TO_BINARY_OP("M_DIV", MNN::MyCustomOpType_M_DIV);
  // TO_BINARY_OP("Equal", MNN::BinaryOpOperation_EQUAL);
  // TO_BINARY_OP("Less", MNN::BinaryOpOperation_LESS);
  // TO_BINARY_OP("LessOrEqual", MNN::BinaryOpOperation_LESS_EQUAL);
  // TO_BINARY_OP("Greater", MNN::BinaryOpOperation_GREATER);
  // TO_BINARY_OP("GreaterOrEqual", MNN::BinaryOpOperation_GREATER_EQUAL);
  // TO_BINARY_OP("Pow", MNN::BinaryOpOperation_POW);
  // TO_BINARY_OP("Sub", MNN::BinaryOpOperation_SUB);
  // TO_BINARY_OP("Or", MNN::BinaryOpOperation_LOGICALOR);
  // TO_BINARY_OP("Xor", MNN::BinaryOpOperation_LOGICALXOR);

  dstOp->main.value = param;
}

REGISTER_CONVERTER(MyCustomOpOnnx, M_ADD);
REGISTER_CONVERTER(MyCustomOpOnnx, M_SUB);
REGISTER_CONVERTER(MyCustomOpOnnx, M_MUL);
REGISTER_CONVERTER(MyCustomOpOnnx, M_DIV);