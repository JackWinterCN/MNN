#include "XPUBackend.hpp"

#include <MNN/MNNForwardType.h>
#include "core/OpCommonUtils.hpp"
#include "backend/xpu/execution/XPUCast.hpp"
#include "backend/xpu/execution/XPUTensorConvert.hpp"

namespace MNN {

std::map<OpType, XPUBackend::OpCreator*> XPUBackend::mOpCreatorsMap;

XPURuntime::XPURuntime(const Backend::Info &info) {
  MNN_PRINT("[XPU] XPURuntime().\n");
  mInfo = info;
  BackendConfig::PrecisionMode precision = BackendConfig::Precision_Normal;
  BackendConfig::PowerMode power = BackendConfig::Power_Normal;
  if (nullptr != mInfo.user) {
    precision = mInfo.user->precision;
    power = mInfo.user->power;
  }
  mPrecision = precision;

  auto rawAlloc = BufferAllocator::Allocator::createDefault();
  mStaticAllocator.reset(new EagerBufferAllocator(rawAlloc));
}

XPURuntime::~XPURuntime() {}

Backend *XPURuntime::onCreate(const BackendConfig *config,
                              Backend *origin) const {
  MNN_PRINT("[XPU] XPURuntime::onCreate().\n");
  return new XPUBackend(this);
}

void XPURuntime::onGabageCollect(int level) {
  // nothing now
}
XPURuntime::CompilerType XPURuntime::onGetCompilerType() const {
  return Compiler_Geometry;
}

REGISTER_RUNTIME_CREATOR(XPURuntime, MNNForwardType::MNN_FORWARD_XPU)


XPUBackend::XPUBackend(const XPURuntime *runtime) : Backend(MNN_FORWARD_XPU) {
  MNN_PRINT("[XPU] XPUBackend().\n");
  mRuntime = runtime;
  mPrecision = mRuntime->mPrecision;

}
XPUBackend::~XPUBackend() {}

Execution *XPUBackend::onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) {
  MNN_PRINT("[XPU] XPUBackend::onCreate().\n");
  auto iter = mOpCreatorsMap.find(op->type());

  if (iter == mOpCreatorsMap.end()) {
    MNN_PRINT("[XPU] Don't support type %s.\n",
              MNN::EnumNameOpType(op->type()));
    return nullptr;
  }

  auto exe = iter->second->onCreate(inputs, outputs, op, this);

  if (nullptr == exe) {
    MNN_PRINT("[XPU] The Creator Don't support type %s.\n",
              MNN::EnumNameOpType(op->type()));
    return nullptr;
  }

  return exe;
}

void XPUBackend::XPUBackend::onExecuteBegin() const {
  MNN_PRINT("[XPU] XPUBackend::onExecuteBegin().\n");
  // mRuntime->pCurrentStatus = mDmaInfo->mDynamicAllocator->apply();
  // if (NO_ERROR != mRuntime->pCurrentStatus) {
  //   return;
  // }
  // if (nullptr != mDmaInfo->mDynamicAllocatorBackup.get()) {
  //   mRuntime->pCurrentStatus = mDmaInfo->mDynamicAllocatorBackup->apply();
  // }
}

void XPUBackend::onExecuteEnd() const {
  MNN_PRINT("[XPU] XPUBackend::onExecuteEnd().\n");
}

void XPUBackend::onResizeBegin() {
    MNN_PRINT("[XPU] XPUBackend::onResizeBegin().\n");
    // mDmaInfo->mCurrentDynamicAllocator->reset();
}

ErrorCode XPUBackend::onResizeEnd() {
  MNN_PRINT("[XPU] XPUBackend::onResizeEnd().\n");
  // auto code = mDmaInfo->mCurrentDynamicAllocator->compute();
  // if (NO_ERROR != code) {
  //   return code;
  // }
  return NO_ERROR;
}

Backend::MemObj *XPUBackend::onAcquire(const Tensor *tensor,
                                       StorageType storageType) {
  MNN_PRINT("[XPU] XPUBackend::onAcquire().\n");

  auto nativeTensor = (Tensor *)tensor;
  auto size = tensor->size();
  Tensor *dest = nativeTensor;
  auto originMem = TensorUtils::getDescribeOrigin(dest)->mem.get();
  if (nullptr != originMem) {
    if (static_cast<XPUMemObj *>(originMem)->getSize() >= size) {
      return originMem;
    } else {
      TensorUtils::getDescribeOrigin(dest)->mem = nullptr;
    }
  }
  // MNN_PRINT("Acquire size = %d\n", size);
  if (size <= 0) {
    MNN_PRINT("Acquire buffer size = %lu\n", size);
    MNN_ASSERT(false);
    return nullptr;
  }
  // if (size > LARGE_MEMORY) {
  //     MNN_PRINT("Size larger than 500 M :%d\n", size);
  // }
  auto &buffer = dest->buffer();
  auto des = TensorUtils::getDescribe(dest);
  MemChunk chunk;
  switch (storageType) {
    case STATIC: {
      chunk = mRuntime->mStaticAllocator->alloc(size, false);
      break;
    }
    case DYNAMIC: {
      // chunk = mDmaInfo->mCurrentDynamicAllocator->alloc(size, false);
      break;
    }
    case DYNAMIC_SEPERATE: {
      // chunk = mDmaInfo->mCurrentDynamicAllocator->alloc(size, true);
      break;
    }
    default:
      MNN_ASSERT(false);
      break;
  }

  if (chunk.invalid()) {
    MNN_ERROR("Alloc buffer error for cpu backend\n");
    return nullptr;
  }

  Backend::MemObj *res = nullptr;

  if (storageType == STATIC) {
    res = new XPUMemObj(mRuntime->mStaticAllocator.get(), chunk, size);
  } else {
    res = new XPUMemObj(mDmaInfo->mCurrentDynamicAllocator, chunk, size);
    chunk.attach(dest);
  }
  if (chunk.ptr()) {
    buffer.host = chunk.ptr();
  }
  des->extra.offset = 0;
  return res;
}

bool XPUBackend::onClearBuffer() {
  MNN_PRINT("[XPU] XPUBackend::onClearBuffer().\n");
  return true;
}

void XPUBackend::onCopyBuffer(const Tensor *srcTensor,
                              const Tensor *dstTensor) const {
  MNN_PRINT("[XPU] XPUBackend::onCopyBuffer().\n");

  auto &srcBuffer = srcTensor->buffer();
  auto &dstBuffer = dstTensor->buffer();
  if (srcBuffer.dimensions != dstBuffer.dimensions) {
    MNN_ERROR("srcBuffer dimension not equal to dstBuffer, can't copy buffer\n");
  }
  if (srcTensor->getDimensionType() == dstTensor->getDimensionType()) {
    for (int i = 0; i < srcBuffer.dimensions; ++i) {
      MNN_ASSERT(srcBuffer.dim[i].extent <= dstBuffer.dim[i].extent);
    }
  }
  if (nullptr == srcBuffer.host || nullptr == dstBuffer.host) {
    return;
  }
  std::unique_ptr<Tensor> wrapTensor;
  if (getDataType(srcTensor) != getDataType(dstTensor)) {
    auto dimType = OpCommonUtils::convertDimType(
        TensorUtils::getDescribe(srcTensor)->dimensionFormat);
    auto convertType = XPUCastCreator::FlOAT_TO_INT8;
    if (getDataType(srcTensor) == DataType_DT_INT8) {
      convertType = XPUCastCreator::INT8_TO_FlOAT;
    }
    wrapTensor.reset(Tensor::createDevice(srcTensor->shape(),
                                          dstTensor->getType(), dimType));
    auto dstType = getDataType(dstTensor);
    if (dstType != DataType_DT_FLOAT) {
      wrapTensor->setType(dstType);
    }
    wrapTensor->buffer().host = (uint8_t *)MNNMemoryAllocAlign(
        wrapTensor->size(),
        MNN_MEMORY_ALIGN_DEFAULT);

#ifdef LOG_VERBOSE
    MNN_PRINT("CPU backend copy tensor ptr:%p -> ptr:%p hostPtr:%p -> %p, "
              "format %d -> %d, dims: [",
              srcTensor, dstTensor, srcTensor->host<void>(),
              dstTensor->host<void>(),
              TensorUtils::getDescribe(srcTensor)->dimensionFormat,
              TensorUtils::getDescribe(dstTensor)->dimensionFormat);
    for (int i = 0; i < srcTensor->dimensions(); ++i) {
      MNN_PRINT("%d ", srcTensor->length(i));
    }
    MNN_PRINT("]\n");
#endif

    TensorUtils::getDescribe(wrapTensor.get())->memoryType =
        Tensor::InsideDescribe::MEMORY_HOST;
    auto code =
        XPUCastCreator::cast(srcTensor, wrapTensor.get(), this, convertType);
    if (NO_ERROR != code) {
      MNN_ERROR("Error in CPUBackend::onCopyBuffer:cast\n");
    }
    srcTensor = wrapTensor.get();
  } else if (srcTensor->getType() != dstTensor->getType()) {
    MNN_ERROR("Input type not match session's tensor\n");
    return;
  }
  auto code = XPUTensorConverter::convert(srcTensor, dstTensor);
  if (NO_ERROR != code) {
    MNN_ERROR("Error in CPUBackend::onCopyBuffer:convert\n");
  }
}

bool XPUBackend::addOpCreator(OpType t, OpCreator *c) {
  MNN_PRINT("[XPU] XPUBackend::addOpCreator(), OpType: %d.\n", t);
  if (mOpCreatorsMap.find(t) != mOpCreatorsMap.end()) {
    MNN_PRINT("Error: %d type has be added\n", t);
    return false;
  }
  mOpCreatorsMap.insert(std::make_pair(t, c));
  return true;
}

DataType XPUBackend::getDataType(const Tensor* tensor) {
    auto des = TensorUtils::getDescribe(tensor);
    if (nullptr == des->quantAttr.get()) {
        return DataType_DT_FLOAT;
    }
    return des->type;
}

}
