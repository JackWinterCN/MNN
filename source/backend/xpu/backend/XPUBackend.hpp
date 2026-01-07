#ifndef MNN_XPU_BACKEND_HPP
#define MNN_XPU_BACKEND_HPP

#include <vector>

#include "XPUMemPool.hpp"
#include <core/Backend.hpp>
#include <core/Execution.hpp>
#include <core/TensorUtils.hpp>
#include "core/BufferAllocator.hpp"
#include "MNN_generated.h"

namespace MNN {

class XPURuntime : public Runtime {
  friend class XPUBackend;

public:
  XPURuntime(const Backend::Info &info);
  virtual ~XPURuntime();
  virtual CompilerType onGetCompilerType() const override;
  virtual Backend *onCreate(const BackendConfig *conf,
                            Backend *origin) const override;
  virtual void onGabageCollect(int level) override;
  virtual std::pair<const void *, size_t> onGetCache() override {
    return std::make_pair(mCacheBuffer, mCacheSize);
  }

private:
  Backend::Info mInfo;
  BackendConfig::PrecisionMode mPrecision;
  const void *mCacheBuffer = nullptr;
  size_t mCacheSize = 0;
  mutable std::shared_ptr<EagerBufferAllocator> mStaticAllocator;
};

class XPUBackend final : public Backend {
public:
  class OpCreator {
  public:
    virtual ~OpCreator() = default;
    virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &output,
                                const MNN::Op *op, Backend *backend) const = 0;
  };

public:
  XPUBackend(const XPURuntime *runtime);
  virtual ~XPUBackend();
  virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const MNN::Op *op) override;
  virtual void onExecuteBegin() const override;
  virtual void onExecuteEnd() const override;
  virtual void onResizeBegin() override;
  virtual ErrorCode onResizeEnd() override;

  virtual MemObj *onAcquire(const Tensor *tensor,
                            StorageType storageType) override;
  virtual bool onClearBuffer() override;
  virtual void onCopyBuffer(const Tensor *srcTensor,
                            const Tensor *dstTensor) const override;
  static bool addOpCreator(OpType t, OpCreator *c);

private:
  float getBytes(const Tensor *tensor);
  void copyFromDevice(const Tensor *srcTensor, const Tensor *dstTensor) const;
  // void copyFromDeviceInt8(const Tensor *srcTensor,
  //                         const Tensor *dstTensor) const;
  void copyToDevice(const Tensor *srcTensor, const Tensor *dstTensor) const;
  // void copyToDeviceInt8(const Tensor *srcTensor, const Tensor *dstTensor) const;
  void copyBetweenDevice(const Tensor *srcTensor,
                         const Tensor *dstTensor) const;

private:
  static std::map<OpType, OpCreator *> mOpCreatorsMap;
  const XPURuntime *mRuntime;
  BackendConfig::PrecisionMode mPrecision;
  std::shared_ptr<XPU::XPUMemPool> mExecutionBufferPool;
};

template <class T>
class XPUOpCreatorRegister {
public:
  XPUOpCreatorRegister(OpType type) {
    T *t = new T;
    XPUBackend::addOpCreator(type, t);
  }
  ~XPUOpCreatorRegister() = default;
};

template <typename T>
class TypedXPUOpCreator : public XPUBackend::OpCreator {
public:
  virtual ~TypedXPUOpCreator() = default;
  virtual Execution *onCreate(const std::vector<Tensor *> &inputs,
                              const std::vector<Tensor *> &outputs,
                              const MNN::Op *op,
                              Backend *backend) const override {
    auto newOp = new T(backend, op, inputs, outputs);
    return newOp;
  }
};

#define REGISTER_XPU_OP_CREATOR(OpClass, opType)                               \
  namespace {                                                                  \
  XPUOpCreatorRegister<TypedXPUOpCreator<OpClass>> _registrar(opType);         \
  }

#define REGISTER_XPU_OP_CUSTOMIZED_CREATOR(OpCreatorClass, opType)             \
  namespace {                                                                  \
  XPUOpCreatorRegister<OpCreatorClass> _registrar(opType);                     \
  }

} // namespace MNN
#endif // MNN_XPU_BACKEND_HPP