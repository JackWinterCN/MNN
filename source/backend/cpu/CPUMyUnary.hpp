#ifndef CPUMYCUSTOMOP_HPP
#define CPUMYCUSTOMOP_HPP

#include "MNN_generated.h"
#include "backend/cpu/CPUBackend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {

class CPUMyUnary : public Execution {
public:
  CPUMyUnary(Backend *bn);
  CPUMyUnary(Backend *bn, int32_t op_type);
  CPUMyUnary(Backend *bn, const MNN::Op *op,
             const std::vector<Tensor *> &inputs,
             const std::vector<Tensor *> &outputs);
  virtual ~CPUMyUnary() = default;
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