#include "XPUBackend.hpp"

#include <MNN/MNNForwardType.h>
#include "core/OpCommonUtils.hpp"
#include "backend/xpu/backend/XPURunningUtils.hpp"
#include "backend/xpu/execution/XPUCast.hpp"
#include "backend/xpu/execution/XPUTensorConvert.hpp"

namespace MNN {
using namespace XPU;
std::map<OpType, XPUBackend::OpCreator*> XPUBackend::mOpCreatorsMap;

XPURuntime::XPURuntime(const Backend::Info &info) {
  MNN_PRINT("[XPU] XPURuntime().\n");
  MNNXPUCoreFunctionInit();
  MNNXPUCoreInt8FunctionInit();
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
  MNN_PRINT("[XPU] XPUBackend(), this: %p\n", this);
  mRuntime = runtime;
  mPrecision = mRuntime->mPrecision;

  mExecutionBufferPool.reset(new XPU::XPUMemPool);
  mCoreFunctions = MNNGetXPUCoreFunctions();
  mInt8CoreFunctions = MNNXPUGetInt8CoreFunctions();
}
XPUBackend::~XPUBackend() {
  mExecutionBufferPool->clear();
  MNN_PRINT("[XPU] XPUBackend::~XPUBackend(), this: %p.\n", this);
}

Execution *XPUBackend::onCreate(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs,
                                const MNN::Op *op) {
  MNN_PRINT("[XPU] XPUBackend::onCreate(), type %s, this: %p.\n",
            MNN::EnumNameOpType(op->type()), this);
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
  MNN_PRINT("[XPU] XPUBackend::onExecuteBegin(), this: %p.\n", this);
}

void XPUBackend::onExecuteEnd() const {
  MNN_PRINT("[XPU] XPUBackend::onExecuteEnd(), this: %p.\n", this);
}

void XPUBackend::onResizeBegin() {
    MNN_PRINT("[XPU] XPUBackend::onResizeBegin(), this: %p.\n", this);
}

ErrorCode XPUBackend::onResizeEnd() {
  MNN_PRINT("[XPU] XPUBackend::onResizeEnd(), this: %p.\n", this);
  return NO_ERROR;
}

size_t XPUBackend::getTensorSize(const Tensor* tensor, bool multiBytes) const {
    auto core = mCoreFunctions;
    size_t dataSize = 1;
    auto des = TensorUtils::getDescribe(tensor);
    for (int i = 0; i < tensor->dimensions(); i++) {
        size_t currentDimSize = tensor->length(i);
        if (des->dimensionFormat == MNN_DATA_FORMAT_NC4HW4 && 1 == i) {
            currentDimSize = UP_DIV(currentDimSize, core->pack) * core->pack;
        }
        dataSize *= currentDimSize;
    }
    if (multiBytes) {
        size_t bytes = tensor->getType().bytes();
        if (TensorUtils::getDescribe(tensor)->quantAttr != nullptr) {
            if (TensorUtils::getDescribe(tensor)->type == DataType_DT_FLOAT) {
                bytes = 4;
            } else {
                bytes = 1;
            }
        }
        return dataSize * bytes;
    }
    return dataSize;
}

size_t XPUBackend::getBytes(const Backend* backend, const Tensor* output) {
    size_t bytes = output->getType().bytes();
    auto core = static_cast<const XPUBackend*>(backend)->functions();
    auto quant = TensorUtils::getDescribe(output)->quantAttr.get();
    if (output->getType().code == halide_type_float) {
        bytes = core->bytes;
    }
    if (nullptr != quant && TensorUtils::getDescribe(output)->type == DataType_DT_INT8) {
        bytes = 1;
    }
    return bytes;
}

Backend::MemObj *XPUBackend::onAcquire(const Tensor *tensor,
                                       StorageType storageType) {
  auto tensorShape = XPU::tensorShapeFormat(tensor);
  int N = tensorShape.at(0);
  int H = tensorShape.at(1);
  int W = tensorShape.at(2);
  int C = tensorShape.at(3);

  size_t size = tensor->size();;
  // if (MNN_DATA_FORMAT_NC4HW4 == TensorUtils::getDescribe(tensor)->dimensionFormat &&
  //     tensor->dimensions() >= 2) {
  //   auto alignC = ROUND_UP(C, 4);
  //   // increment of height and width
  //   auto hR = ROUND_UP(H + 3, 4) - H;
  //   auto wR = ROUND_UP(W + 3, 4) - W;
  //   size = N * alignC * W * H;
  //   size = size + hR * W * 4 + wR * 4;
  // } else {
  //   size = N * H * W * C;
  //   size = ROUND_UP(size, 4);
  // }
  // size = ROUND_UP(size, 2);
  float typeSize = getBytes(tensor);

  MNN_PRINT("[XPU] XPUBackend::onAcquire() storageType:%d, NHWC:[%d, %d, %d, %d], "
            "typeSize: %f, size: %ld, this: %p\n",
            storageType, N, H, W, C, typeSize, size, this);

  if (storageType != STATIC && storageType != DYNAMIC) {
    MNN_ERROR("not support storageType %d\n", storageType);
    return nullptr;
  }
  auto node = mExecutionBufferPool->alloc(size);
  ((Tensor *)tensor)->buffer().device = reinterpret_cast<uint64_t>(node->physical_addr);
  return new XPU::XPUDeviceMemObj(node, mExecutionBufferPool.get());
}

const Runtime* XPUBackend::getRuntime() {
    return mRuntime;
}

bool XPUBackend::onClearBuffer() {
  MNN_PRINT("[XPU] XPUBackend::onClearBuffer(), this: %p.\n", this);
  return true;
}

void XPUBackend::copyFromDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    auto needSize = dstTensor->size();
    auto shape = XPU::tensorShapeFormat(srcTensor);
    auto srcDimensionFormat = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto dstDimensionFormat = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto memType = dstTensor->buffer().flags;
    bool directCopy = (srcDimensionFormat == dstDimensionFormat || srcTensor->dimensions() <= 1)
                      //  && MNN::MNN_DATA_FORMAT_NC4HW4 != dstDimensionFormat && MNN_DATA_FORMAT_NC4HW4 != srcDimensionFormat
                       && (XPU::getDataType(srcTensor) == XPU::getDataType(dstTensor));
    // if (mPrecision != BackendConfig::Precision_High) { // Fp16
    //     if (dstTensor->getType().code == halide_type_float) {
    //         directCopy = false;
    //     }
    // }

    if (directCopy) {
      void *hostPtr = dstTensor->host<float>();
      memcpy(hostPtr, (void *)srcTensor->deviceId(), needSize);
      MNN_PRINT("[XPU] (device->host) size: %d bytes, %p ->%p\n", needSize,
                (void *)srcTensor->deviceId(), hostPtr);
      return;
    } else {
      MNN_ERROR("copyFromDevice can NOT direct copy, srcDimensionFormat:%d, "
                "dstDimensionFormat:%d, dataType:%d\n",
                srcDimensionFormat, dstDimensionFormat, XPU::getDataType(srcTensor));
    }
}

void XPUBackend::copyToDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
    auto needSize = srcTensor->size();
    auto shape = XPU::tensorShapeFormat(srcTensor);
    auto srcDimensionFormat = TensorUtils::getDescribe(srcTensor)->dimensionFormat;
    auto dstDimensionFormat = TensorUtils::getDescribe(dstTensor)->dimensionFormat;
    auto memType = srcTensor->buffer().flags;
    void* hostPtr = srcTensor->host<float>();

    bool directCopy = (srcDimensionFormat == dstDimensionFormat || srcTensor->dimensions() <= 1)
                      //  && MNN_DATA_FORMAT_NC4HW4 != dstDimensionFormat && MNN_DATA_FORMAT_NC4HW4 != srcDimensionFormat
                       && (XPU::getDataType(srcTensor) == XPU::getDataType(dstTensor));
    // if (mPrecision != BackendConfig::Precision_High) { // Fp16
    //     if (dstTensor->getType().code == halide_type_float) {
    //         directCopy = false;
    //     }
    // }
    if(directCopy){
      memcpy((void *)dstTensor->deviceId(), hostPtr, needSize);
      MNN_PRINT("[XPU] (host->device) size: %d bytes, %p -> %p\n", needSize,
                hostPtr, (void *)dstTensor->deviceId());
      return;
    } else {
      MNN_ERROR("copyToDevice can NOT direct copy, srcDimensionFormat:%d, "
                "dstDimensionFormat:%d, dataType:%d\n",
                srcDimensionFormat, dstDimensionFormat, XPU::getDataType(srcTensor));
    }
}

// void XPUBackend::copyBetweenDevice(const Tensor* srcTensor, const Tensor* dstTensor) const{
//     int srcMemtype = srcTensor->buffer().flags;
//     int dstMemtype = dstTensor->buffer().flags;
//     if(MNN_FORWARD_CPU == srcMemtype && MNN_FORWARD_CPU == dstMemtype){
//         mCLRuntime->copyBetweenDevice(srcTensor, dstTensor, mPrecision, mMemType);
//     } else {
//         const Tensor* hostTensor = MNN_FORWARD_CPU != srcMemtype ? srcTensor : dstTensor;
//         const Tensor* deviceTensor = MNN_FORWARD_CPU == srcMemtype ? srcTensor : dstTensor;
//         MNN_DATA_FORMAT data_format = TensorUtils::getDescribe(deviceTensor)->dimensionFormat;

//         bool alloc_error = _allocHostBuffer(0, hostTensor);
//         if(false == alloc_error){
//             MNN_ERROR("Alloc _allocHostBuffer error\n");
//             return;
//         }

//         //Covert format
//         if(MNN_FORWARD_CPU != srcMemtype){
//             mCLRuntime->convertToDevice(hostTensor, deviceTensor, data_format, mPrecision, mMemType, false, srcMemtype);
//         }else{
//             mCLRuntime->convertFromDevice(deviceTensor, hostTensor, data_format, mPrecision, mMemType, false, dstMemtype);
//         }
//     }
// }

// void CLRuntime::copyBetweenDevice(const Tensor* srcTensor, const Tensor* dstTensor, int precision, int backend_memtype) const{
//     int input_precision = ((OpenCLBackend*)(TensorUtils::getDescribeOrigin(srcTensor)->getBackend()))->getPrecision();
//     int output_precision = ((OpenCLBackend*)(TensorUtils::getDescribeOrigin(dstTensor)->getBackend()))->getPrecision();
//     #ifndef MNN_OPENCL_BUFFER_CLOSED
//     if(backend_memtype == BUFFER)
//     {
//         OpenCL::convertBufferToBuffer(const_cast<Tensor*>(srcTensor), const_cast<Tensor*>(dstTensor), mOpenCLRuntime.get(), input_precision, output_precision, precision, true, true);
//     }
//     else
//     #endif /* MNN_OPENCL_BUFFER_CLOSED */
//     if(input_precision == output_precision){
//         std::vector<int> bufferShape = MNN::OpenCL::tensorShapeFormat(srcTensor);

//         mOpenCLRuntime.get()->commandQueue().enqueueCopyImage(
//                 openCLImage(srcTensor), openCLImage(dstTensor),
//                 {0, 0, 0}, {0, 0, 0},
//                 {(size_t)bufferShape[2]* UP_DIV(bufferShape[3], 4), (size_t)bufferShape[0]*bufferShape[1], 1});
//     } else{
//         OpenCL::convertImageToImage(const_cast<Tensor*>(srcTensor), const_cast<Tensor*>(dstTensor), mOpenCLRuntime.get(), input_precision, output_precision, precision);
//     }
//     return;
// }


// void OpenCLBackend::onCopyBuffer(const Tensor* srcTensor, const Tensor* dstTensor) const {
//     clearRecord();
//     if (srcTensor->host<float>() != nullptr) {
//         copyToDevice(srcTensor, dstTensor);
//     }else if(dstTensor->host<void>() != nullptr){
//         copyFromDevice(srcTensor, dstTensor);
//     }else{
//         copyBetweenDevice(srcTensor, dstTensor);
//     }

// }

void XPUBackend::onCopyBuffer(const Tensor *srcTensor,
                              const Tensor *dstTensor) const {
  MNN_PRINT("[XPU] XPUBackend::onCopyBuffer(), this: %p.\n", this);
  // if (srcTensor->host<float>() == nullptr) {
  //   MNN_PRINT("srcTensor host: null. ");
  // }
  // if (dstTensor->host<void>() == nullptr) {
  //   MNN_PRINT("dstTensor host: null. \n");
  // }
  if (srcTensor->host<float>() != nullptr) {
    copyToDevice(srcTensor, dstTensor);
  } else if (dstTensor->host<void>() != nullptr) {
    copyFromDevice(srcTensor, dstTensor);
  } else {
    MNN_PRINT("copyBetweenDevice[Not supported]\n");
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

float XPUBackend::getBytes(const Tensor* tensor) {
    float bytes = (float)tensor->getType().bytes();
    if (mPrecision != BackendConfig::Precision_High) {// Fp16
        if (halide_type_float == tensor->getType().code) {
            bytes = 2.0;
        }
    }
    auto quant = TensorUtils::getDescribe(tensor)->quantAttr.get();
    if (nullptr != quant && TensorUtils::getDescribe(tensor)->type == DataType_DT_INT8) {
        bytes = 1.0;
    }
    if(tensor->getType().bits == 4) {
        bytes = 0.5;
    }
    return bytes;
}

}
