#ifndef XPUMYCUSTOMOP_HPP
#define XPUMYCUSTOMOP_HPP

#include "MNN_generated.h"
#include "backend/xpu/backend/XPUBackend.hpp"
#include "core/AutoStorage.h"
#include "core/Execution.hpp"

namespace MNN {

class XPUMyCustomOp : public Execution {
public:
  XPUMyCustomOp(Backend *bn);
  XPUMyCustomOp(Backend *bn, int32_t op_type);
  XPUMyCustomOp(Backend *bn, const MNN::Op *op,
                const std::vector<Tensor *> &inputs,
                const std::vector<Tensor *> &outputs);
  virtual ~XPUMyCustomOp() = default;

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