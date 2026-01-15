#ifndef XPU_MY_UNARY_HPP
#define XPU_MY_UNARY_HPP

#include "MNN_generated.h"
#include "backend/xpu/backend/XPUBackend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {

class XPUMyUnary : public Execution {
public:
  XPUMyUnary(Backend *bn);
  XPUMyUnary(Backend *bn, int32_t op_type);
  XPUMyUnary(Backend *bn, const MNN::Op *op,
             const std::vector<Tensor *> &inputs,
             const std::vector<Tensor *> &outputs);
  virtual ~XPUMyUnary() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

private:
  int mRealSize;
  int mOperand{0};
  int32_t myUnaryType = MyCustomUnaryFuncType_M_UNARY_ABS;
};

} // namespace MNN
#endif