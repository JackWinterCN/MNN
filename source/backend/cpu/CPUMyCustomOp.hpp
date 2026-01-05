#ifndef CPUMYCUSTOMOP_HPP
#define CPUMYCUSTOMOP_HPP

#include "backend/cpu/CPUBackend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"
#include "MNN_generated.h"

namespace MNN {

class CPUMyCustomOp : public Execution {
public:
  CPUMyCustomOp(Backend *bn) : Execution(bn) {}
  CPUMyCustomOp(Backend *bn, int32_t op_type)
      : Execution(bn), myCustomOpType(op_type) {}
  virtual ~CPUMyCustomOp() = default;

  // 若执行onExecute需要使用缓存，在此函数中申请，若无可不声明
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
  // 具体的Op执行函数
  virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs) override;

private:
  AutoStorage<uint8_t> mOffset;
  AutoStorage<uint8_t> mCacheSrc;
  AutoStorage<uint8_t> mCacheDst;
  int mRealSize;
  int32_t myCustomOpType = MyCustomOpType_M_ADD;
};

} // namespace MNN
#endif