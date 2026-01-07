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

class XPUMemObj : public Backend::MemObj {
public:
  XPUMemObj(BufferAllocator *allocator, MemChunk chunk, int size)
      : mAllocator(allocator), mChunk(chunk), mSize(size) {}
  virtual ~XPUMemObj() {
    if (mAllocator) {
      mAllocator->free(mChunk);
    }
  }
  virtual MemChunk chunk() { return mChunk; }
  inline int getSize() const { return mSize; }

private:
  BufferAllocator *mAllocator;
  MemChunk mChunk;
  int mSize;
};

class XPURuntime : public Runtime {
public:
  struct DynamicAllocator {
    std::shared_ptr<BufferAllocator> mDynamicAllocator;
    std::shared_ptr<BufferAllocator> mDynamicAllocatorBackup;
    BufferAllocator *mCurrentDynamicAllocator = nullptr;
  };

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
  // mutable IHostMemory* mModel = nullptr;
  const void *mCacheBuffer = nullptr;
  size_t mCacheSize = 0;
  mutable std::shared_ptr<EagerBufferAllocator> mStaticAllocator;
  mutable std::shared_ptr<DynamicAllocator> mSharedDmaInfo;

  friend class XPUBackend;
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
  static DataType getDataType(const Tensor* tensor);
  static bool addOpCreator(OpType t, OpCreator *c);
private:
    float getBytes(const Tensor* tensor);
    void copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
    // void copyFromDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;
    // void copyToDeviceInt8(const Tensor* srcTensor, const Tensor* dstTensor) const;
    void copyBetweenDevice(const Tensor* srcTensor, const Tensor* dstTensor) const;
private:
  static std::map<OpType, OpCreator *> mOpCreatorsMap;
  const XPURuntime *mRuntime;
  BackendConfig::PrecisionMode mPrecision;
  std::shared_ptr<XPURuntime::DynamicAllocator> mDmaInfo;
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


// #define REGISTER_XPU_OP_CREATOR_BY_MACRO(OpClass, opType)                      \
//   namespace {                                                                  \
//   class OpClass##Creator : public XPUBackend::Creator {                        \
//   public:                                                                      \
//     virtual ~##OpClass##Creator() = default;                                   \
//     virtual Execution *onCreate(const std::vector<Tensor *> &inputs,           \
//                                 const std::vector<Tensor *> &outputs,          \
//                                 const MNN::Op *op,                             \
//                                 Backend *backend) const override {             \
//       auto newOp = new OpClass(backend, op, inputs, outputs);                  \
//       return newOp;                                                            \
//     }                                                                          \
//   };                                                                           \
//   class OpClass##CreatorRegister {                                             \
//   public:                                                                      \
//     OpClass##CreatorRegister() {                                               \
//       OpClass##Creator *t = new OpClass##Creator;                              \
//       XPUBackend::addCreator(opType, t);                                       \
//     }                                                                          \
//     ~OpClass##CreatorRegister() = default;                                     \
//   };                                                                           \
//   OpClass##CreatorRegister _registrar();                                       \
//   }

inline std::vector<int> tensorShapeFormat(const Tensor *input) {

    int iN = (0 != input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
    int iC = (0 != input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
    int iH = (0 != input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
    int iW = (0 != input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;

    if(input->buffer().dimensions > 4)//more than 4 dimensions put to N dimension
    {
        for(int i = 4; i < input->buffer().dimensions; i++)
        {
            iW *= input->buffer().dim[i].extent;
        }
    }

    if (TensorUtils::getDescribe(input)->dimensionFormat == MNN::MNN_DATA_FORMAT_NHWC)
    {
        iN = (0 < input->buffer().dim[0].extent) ? input->buffer().dim[0].extent : 1;
        iH = (0 < input->buffer().dim[1].extent) ? input->buffer().dim[1].extent : 1;
        iW = (0 < input->buffer().dim[2].extent) ? input->buffer().dim[2].extent : 1;
        iC = (0 < input->buffer().dim[3].extent) ? input->buffer().dim[3].extent : 1;

        if(input->buffer().dimensions > 4)//more than 4 dimensions put to N dimension
        {
            for(int i = 4; i < input->buffer().dimensions; i++)
            {
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

} // namespace MNN
#endif // MNN_XPU_BACKEND_HPP