//
//  XPUDenseConvolutionTiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//
#include "XPUDenseConvolutionTiledExecutor.hpp"

#include <math.h>

#include <MNN/AutoTime.hpp>
#include "XPUCommonOptFunction.hpp"
#include "backend/xpu/execution/compute/Concurrency.h"
#include "core/BufferAllocator.hpp"
#include "core/Macro.h"
#include "core/MemoryFormater.h"
#include "core/TensorUtils.hpp"
#include "math/Vec.hpp"
#include "backend/xpu/execution/XPUTensorConvert.hpp"

#define PARAMETERSIZE 7

using Vec4 = MNN::Math::Vec<float, 4>;
namespace MNN {
namespace XPU {
void XPUDenseConvolutionTiledExecutor::initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const XPUCoreFunctions* function) {
    XPUConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForMatMul_B(dest, cache, outputCount, kernelSize * depth, true);

}
bool XPUDenseConvolutionTiledExecutor::initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<XPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes) {
    int weightLength = hU * lU * hP * lP;
    resource->mDequantize.bits = 8;
    resource->lU = lU;
    resource->hU = hU;
    resource->lP = lP;
    resource->hP = hP;
    MNN_ASSERT(lP == 1);
    // Save scale bias
    int dequantCnt = int8Info->alpha.size();
    int scaleSize = dequantCnt; // real size
    if (int8Info->asymmetric) {
        scaleSize = dequantCnt / 2;

    }
    int blockNum = scaleSize / outputCount;
    scaleSize = blockNum * hU * hP; // pack size
    resource->mDequantize.mScaleBias.reset(MNN::Tensor::createDevice<uint8_t>({scaleSize * 2 * bytes}));
    auto res = resource->backend->onAcquireBuffer(resource->mDequantize.mScaleBias.get(), Backend::STATIC);
    if (!res) {
        return false;
    }
    int originOffset = 0;
    auto srcWInt8 = int8Info->weight.get();
    std::vector<int8_t> blob;
    if (int8Info->canUseInt4) {
        // Revert int4 to int8
        auto size = int8Info->weight.size();
        blob.resize(int8Info->weight.size() * 2);
        auto idxBuf = (uint8_t*)srcWInt8;
        for (int i=0; i<size; ++i) {
            int val = idxBuf[i];
            int x1 = val / 16;
            int x2 = val % 16;
            blob[2 * i] = x1 - 8;
            blob[2 * i + 1] = x2 - 8;

        }
        srcWInt8 = blob.data();
    }
    {
        resource->mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{hU, lU * lP, hP}));
        auto res = resource->backend->onAcquireBuffer(resource->mWeight.get(), Backend::STATIC);
        if (!res) {
            return false;
        }
        // Reorder weight for int8
        auto dstWInt8 = resource->mWeight->host<int8_t>();
        ::memset(dstWInt8, 0, resource->mWeight->usize());
        for (int y=0; y<outputCount; ++y) {
            int yo = y / hP;
            int yi = y % hP;
            auto srcY = srcWInt8 + y * srcChannel * kernelSize;
            auto dstY = dstWInt8 + yo * lP * hP * lU + yi;
            for (int iz=0; iz<srcChannel; ++iz) {
                for (int k=0; k<kernelSize; ++k) {
                    int sx = iz * kernelSize + k;
                    int dx = iz + k * srcChannel;
                    dstY[dx * hP] = srcY[sx];
                }
            }
        }
    }
    auto alphaPtr = resource->mDequantize.mScaleBias->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * bytes);
    ::memset(alphaPtr, 0, 2 * scaleSize * bytes);
    int h = int8Info->alpha.size();
    if (bytes == 2) {
        auto core = static_cast<XPUBackend*>(resource->backend)->functions();
        std::vector<float> tmpAlpha(scaleSize * 2, 0.0f);
        if (int8Info->asymmetric) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = tmpAlpha.data() + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[2 * scaleIndex + 1];
                    dstAlpha[j + scaleSize] = srcAlpha[2 * scaleIndex] + (float)originOffset * dstAlpha[j];
                }
            }
        } else {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = tmpAlpha.data() + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[scaleIndex];
                    dstAlpha[j + scaleSize] = (float)originOffset * dstAlpha[j];
                }
            }
        }
        core->MNNFp32ToLowp(tmpAlpha.data(), reinterpret_cast<int16_t*>(alphaPtr), scaleSize * 2);
    } else {
        if (int8Info->asymmetric) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = alphaPtr + i * hU * hP;
                auto dstBias  = biasPtr + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[2 * scaleIndex + 1];
                    dstBias[j] = srcAlpha[2 * scaleIndex] + (float)originOffset * dstAlpha[j];
                }
            }
        } else {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = alphaPtr + i * hU * hP;
                auto dstBias  = biasPtr + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[scaleIndex];
                    dstBias[j] = (float)originOffset * dstAlpha[j];
                }
            }
        }
    }
    return true;
}

void XPUDenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(lowMemoryMatmulUnit* matmulUnit, lowMemoryMatmulRemain* matmulRemain, float* weightBytes, int32_t weightQuantBits, const XPUCoreFunctions* core) {
    if (weightQuantBits == 8) {
        *matmulUnit = core->MNNPackedMatMul_int8;
        *matmulRemain = core->MNNPackedMatMulRemain_int8;
        *weightBytes  = 1;
    }
}

XPUDenseConvolutionTiledExecutor::XPUDenseConvolutionTiledExecutor(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> int8Info)
    : XPUConvolutionTiledExecutor(b, bias, biasSize) {

    auto outputCount = (int)biasSize;
    int eP, lP, hP;
    auto core = static_cast<XPUBackend*>(b)->functions();
    int bytes = core->bytes;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    bool useInt8Weight = 0 == originWeightSize;
    if (useInt8Weight) {
        MNN_ASSERT(nullptr != int8Info.get());
        originWeightSize = int8Info->weight.size();
    }
    if (int8Info && int8Info->canUseInt4) {
        originWeightSize *= 2;
    }
    // Don't use common->inputCount for old model common->inputCount is zero
    auto srcCount    = (int)originWeightSize / outputCount / common->kernelX() / common->kernelY();
    auto lSize = srcCount * common->kernelX() * common->kernelY();
    auto hU = UP_DIV(outputCount, hP);
    auto lU = UP_DIV(lSize, lP);
    if (useInt8Weight) {
        // Quantize weight to int8
        auto allocSuccess = XPUDenseConvolutionTiledExecutor::initQuantizeResource(int8Info, mResource, hU, hP, lU, lP, outputCount, srcCount, common->kernelX() * common->kernelY(), bytes);
        if (!allocSuccess) {
            mValid = false;
            return;
        }
    } else {
        if (core->matmulBytes != 0) {
            bytes = core->matmulBytes;
        }
        mResource->mWeight.reset(Tensor::createDevice<uint8_t>(
            {hU * hP, lU * lP, bytes}));
        mValid = mValid && backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        std::shared_ptr<Tensor> cache(Tensor::createDevice<uint8_t>({outputCount, srcCount * common->kernelX() * common->kernelY(), (int)sizeof(float)})); // cache must be float
        mValid = mValid && backend()->onAcquireBuffer(cache.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        initWeight((float*)mResource->mWeight->deviceId(), originWeight, (float*)cache->deviceId(), srcCount, outputCount, common->kernelX() * common->kernelY(), core);
        // MNN_PRINT("srcCount:%d, outputCount:%d, dense weight matrix tile:", srcCount, outputCount);
        // formatMatrix(mResource->mWeight->host<float>(), {UP_DIV(outputCount, hP), lSize, hP});
        backend()->onReleaseBuffer(cache.get(), Backend::STATIC);
    }
    mProxy.reset(new DenseConvolutionTiledImpl(common, b, mResource.get()));
}

XPUDenseConvolutionTiledExecutor::XPUDenseConvolutionTiledExecutor(std::shared_ptr<XPUConvolution::Resource> res, const Convolution2DCommon* common, Backend* b) : XPUConvolutionTiledExecutor(res, b) {
    mProxy.reset(new DenseConvolutionTiledImpl(common, b, mResource.get()));
}

XPUDenseConvolutionTiledExecutor::~XPUDenseConvolutionTiledExecutor() {
    // Do nothing
}
bool XPUDenseConvolutionTiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dense = new XPUDenseConvolutionTiledExecutor(mResource, op->main_as_Convolution2D()->common(), bn);
    dense->mProxy->mConvPerfconfig = mProxy->mConvPerfconfig;
    *dst = dense;
    return true;
}

ErrorCode XPUDenseConvolutionTiledExecutor::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    auto code = mProxy->onExecute(mInputs, outputs);
    return code;
}
ErrorCode XPUDenseConvolutionTiledExecutor::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
    auto code = mProxy->onResize(mInputs, outputs);
    if (NO_ERROR != code) {
        return code;
    }
    return NO_ERROR;
}

XPUDenseConvolutionGeneralExecutor::XPUDenseConvolutionGeneralExecutor(const Convolution2DCommon* common, Backend* b,
                                                   const float* originWeight, size_t originWeightSize,
                                                   const float* bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common> int8Info)
    : XPUConvolutionTiledExecutor(b, bias, biasSize) {

    conv_common_param_.padX        = common->padX();
    conv_common_param_.padY        = common->padY();
    conv_common_param_.padMode     = common->padMode();
    conv_common_param_.kernelX     = common->kernelX();
    conv_common_param_.kernelY     = common->kernelY();
    conv_common_param_.strideX     = common->strideX();
    conv_common_param_.strideY     = common->strideY();
    conv_common_param_.dilateX     = common->dilateX();
    conv_common_param_.dilateY     = common->dilateY();
    conv_common_param_.group       = common->group();
    conv_common_param_.outputCount = common->outputCount();
    conv_common_param_.inputCount  = common->inputCount();
    conv_common_param_.relu       = common->relu();
    conv_common_param_.relu6      = common->relu6();
    if (common->pads()) {
        conv_common_param_.pads.assign(common->pads()->begin(), common->pads()->end());
    }
    if (common->outPads()) {
        conv_common_param_.outPads.assign(common->outPads()->begin(), common->outPads()->end());
    }
    conv_common_param_.hasOutputShape = common->hasOutputShape();
   

    conv_bias_.assign(bias, bias + biasSize);
    conv_weight_.assign(originWeight, originWeight + originWeightSize);

    auto outputCount = (int)biasSize;
    int eP, lP, hP;
    auto core = static_cast<XPUBackend*>(b)->functions();
    int bytes = core->bytes;
    core->MNNGetMatMulPackMode(&eP, &lP, &hP);
    bool useInt8Weight = 0 == originWeightSize;
    if (useInt8Weight) {
        MNN_ASSERT(nullptr != int8Info.get());
        originWeightSize = int8Info->weight.size();
    }
    if (int8Info && int8Info->canUseInt4) {
        originWeightSize *= 2;
    }
    // Don't use common->inputCount for old model common->inputCount is zero
    auto srcCount    = (int)originWeightSize / outputCount / common->kernelX() / common->kernelY();
    auto lSize = srcCount * common->kernelX() * common->kernelY();
    auto hU = UP_DIV(outputCount, hP);
    auto lU = UP_DIV(lSize, lP);
    if (useInt8Weight) {
        // Quantize weight to int8
        auto allocSuccess = XPUDenseConvolutionGeneralExecutor::initQuantizeResource(int8Info, mResource, hU, hP, lU, lP, outputCount, srcCount, common->kernelX() * common->kernelY(), bytes);
        if (!allocSuccess) {
            mValid = false;
            return;
        }
    } else {
        if (core->matmulBytes != 0) {
            bytes = core->matmulBytes;
        }
        mResource->mWeight.reset(Tensor::createDevice<uint8_t>(
            {hU * hP, lU * lP, bytes}));
        mValid = mValid && backend()->onAcquireBuffer(mResource->mWeight.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        std::shared_ptr<Tensor> cache(Tensor::createDevice<uint8_t>({outputCount, srcCount * common->kernelX() * common->kernelY(), (int)sizeof(float)})); // cache must be float
        mValid = mValid && backend()->onAcquireBuffer(cache.get(), Backend::STATIC);
        if (!mValid) {
            return;
        }
        initWeight((float*)mResource->mWeight->deviceId(), originWeight, (float*)cache->deviceId(), srcCount, outputCount, common->kernelX() * common->kernelY(), core);
        // MNN_PRINT("srcCount:%d, outputCount:%d, dense weight matrix tile:", srcCount, outputCount);
        // formatMatrix(mResource->mWeight->host<float>(), {UP_DIV(outputCount, hP), lSize, hP});
        backend()->onReleaseBuffer(cache.get(), Backend::STATIC);
    }
    mProxy.reset(new DenseConvolutionTiledImpl(common, b, mResource.get()));
}

XPUDenseConvolutionGeneralExecutor::XPUDenseConvolutionGeneralExecutor(std::shared_ptr<XPUConvolution::Resource> res, const Convolution2DCommon* common, Backend* b) : XPUConvolutionTiledExecutor(res, b) {
    mProxy.reset(new DenseConvolutionTiledImpl(common, b, mResource.get()));
}

XPUDenseConvolutionGeneralExecutor::~XPUDenseConvolutionGeneralExecutor() {
    // Do nothing
}
bool XPUDenseConvolutionGeneralExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (!mValid) {
        return false;
    }
    if (nullptr == dst) {
        return true;
    }
    auto dense = new XPUDenseConvolutionGeneralExecutor(mResource, op->main_as_Convolution2D()->common(), bn);
    dense->mProxy->mConvPerfconfig = mProxy->mConvPerfconfig;
    *dst = dense;
    return true;
}

// 计算输出特征图的高度和宽度
// inputH: 输入特征图高度, inputW: 输入特征图宽度, params: 卷积参数
std::pair<int, int> calculateOutputSize(int inputH, int inputW,
                                        const Convolution2DCommonT &params) {
  // 核心公式：OH = ((IH + 2*padY - kernelY) / strideY) + 1
  int outputH = (inputH + 2 * params.padY - params.kernelY) / params.strideY + 1;
  // 核心公式：OW = ((IW + 2*padX - kernelX) / strideX) + 1
  int outputW = (inputW + 2 * params.padX - params.kernelX) / params.strideX + 1;
  return {outputH, outputW};
}

// 对输入特征图进行零填充（CAFFE模式默认零填充）
// input: 输入特征图 (N, C, H, W)，返回填充后的特征图 (N, C, H+2*padY, W+2*padX)
std::vector<float> padInput(const std::vector<float> &input, int N, int C,
                            int H, int W, const Convolution2DCommonT &params) {
  int padLeft = params.pads[0];
  int padTop = params.pads[1];
  int padRight = params.pads[2];
  int padBottom = params.pads[3];

  // 填充后的高度和宽度
  int paddedH = H + padTop + padBottom;
  int paddedW = W + padLeft + padRight;

  // 初始化填充后的特征图，默认值为0（零填充）
  std::vector<float> paddedInput(N * C * paddedH * paddedW, 0.0f);

  // 将原始输入拷贝到填充后的中心位置
  for (int n = 0; n < N; ++n) {       // 遍历批次
    for (int c = 0; c < C; ++c) {     // 遍历通道
      for (int h = 0; h < H; ++h) {   // 遍历高度
        for (int w = 0; w < W; ++w) { // 遍历宽度
          // 原始输入的一维索引
          int inputIdx = n * C * H * W + c * H * W + h * W + w;
          // 填充后对应的索引（偏移padTop和padLeft）
          int paddedIdx = n * C * paddedH * paddedW + c * paddedH * paddedW +
                          (h + padTop) * paddedW + (w + padLeft);
          paddedInput[paddedIdx] = input[inputIdx];
        }
      }
    }
  }

  return paddedInput;
}

// 卷积算子核心实现
// input: 输入特征图 (N, inputC, H, W)
// weight: 卷积核 (outputC, inputC/group, kernelY, kernelX)
// bias: 偏置 (outputC)，为空则不使用偏置
// N: 批次大小, inputH/inputW: 输入特征图高/宽
std::vector<float> conv2d(const std::vector<float> &input,
                          const std::vector<float> &weight,
                          const std::vector<float> &bias, int N, int inputH,
                          int inputW, const Convolution2DCommonT &params) {
  // 提取核心参数
  int inputC = params.inputCount;
  int outputC = params.outputCount;
  int kernelY = params.kernelY;
  int kernelX = params.kernelX;
  int strideY = params.strideY;
  int strideX = params.strideX;
  int group = params.group;

  // 基础校验：输入/输出通道数需能被分组数整除
  if (inputC % group != 0 || outputC % group != 0) {
    MNN_PRINT("error: inputC:%d, outputC:%d, group:%d", inputC, outputC, group);
    return {};
  }
  int groupInputC = inputC / group;   // 每个分组的输入通道数
  int groupOutputC = outputC / group; // 每个分组的输出通道数

  // 计算输出特征图尺寸
  auto outputSize = calculateOutputSize(inputH, inputW, params);
  int outputH = outputSize.first;
  int outputW = outputSize.second;
  MNN_PRINT("outputH:%d, outputW:%d\n", outputH, outputW);

  // 对输入进行零填充
  std::vector<float> paddedInput =
      padInput(input, N, inputC, inputH, inputW, params);
  int paddedH = inputH + params.padY * 2;
  int paddedW = inputW + params.padX * 2;

  // 初始化输出特征图（默认值0）
  std::vector<float> output(N * outputC * outputH * outputW, 0.0f);

  // 核心卷积计算逻辑
  for (int n = 0; n < N; ++n) {            // 遍历批次
    for (int oc = 0; oc < outputC; ++oc) { // 遍历输出通道
      int g = oc / groupOutputC; // 当前输出通道所属分组（group=1时恒为0）
      for (int oh = 0; oh < outputH; ++oh) {   // 遍历输出高度
        for (int ow = 0; ow < outputW; ++ow) { // 遍历输出宽度
          float sum = 0.0f;                    // 卷积累加和

          // 遍历当前分组内的输入通道
          for (int ic = 0; ic < groupInputC; ++ic) {
            int realIC = g * groupInputC + ic; // 实际输入通道索引

            // 遍历卷积核
            for (int ky = 0; ky < kernelY; ++ky) {   // 核高度
              for (int kx = 0; kx < kernelX; ++kx) { // 核宽度
                // 计算填充后输入的对应位置
                int ih = oh * strideY + ky; // 输入高度位置
                int iw = ow * strideX + kx; // 输入宽度位置

                // 边界检查（填充后可省略，保留更健壮）
                if (ih < 0 || ih >= paddedH || iw < 0 || iw >= paddedW) {
                  continue;
                }

                // 计算输入特征图的一维索引
                int inputIdx = n * inputC * paddedH * paddedW +
                               realIC * paddedH * paddedW + ih * paddedW + iw;

                // 计算卷积核的一维索引
                int weightIdx = oc * groupInputC * kernelY * kernelX +
                                ic * kernelY * kernelX + ky * kernelX + kx;

                // 累加：输入值 × 权重值
                sum += paddedInput[inputIdx] * weight[weightIdx];
              }
            }
          }

          // 加上偏置（如果有）
          if (!bias.empty()) {
            sum += bias[oc];
          }

          // 激活函数（当前配置无激活，直接赋值）
          if (params.relu)
            sum = std::max(0.0f, sum);
          if (params.relu6)
            sum = std::min(std::max(0.0f, sum), 6.0f);

          // 赋值到输出特征图
          int outputIdx = n * outputC * outputH * outputW +
                          oc * outputH * outputW + oh * outputW + ow;
          output[outputIdx] = sum;
        }
      }
    }
  }

  return output;
}


template<typename T>
void MNNPackC4Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[4];
    const T* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        auto dstZ = dst + z * areaOffset[1] * 4;
        for(y = 0; y < 4; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < 4; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
        }
        srcOffset += areaOffset[0] * 4;
    }
    if(remain > 0){
        auto dstZ = dst + depthC4 * areaOffset[1] * 4;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + areaOffset[0] * y;
        }
        for(x = 0; x < area; ++x) {
            for(y = 0; y < remain; ++y) {
                dstZ[0] = srcChannel[y][x];
                dstZ++;
            }
            for(y = remain; y < 4; ++y) {
                dstZ[0] = 0;
                dstZ++;
            }
        }
    }
}

ErrorCode XPUDenseConvolutionGeneralExecutor::onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    ErrorCode code = NO_ERROR;
    // auto code = mProxy->onExecute(mInputs, outputs);

    Tensor input_nchw;
    TensorUtils::copyShape(inputs[0], &input_nchw, true, true);
    TensorUtils::getDescribe(&input_nchw)->dimensionFormat = MNN::MNN_DATA_FORMAT_NCHW;
    backend()->onAcquireBuffer(&input_nchw, Backend::STATIC);
    MNN::XPUTensorConverter::convert(inputs[0], &input_nchw);
    conv_input_.resize(input_nchw.elementSize());
    memcpy((void*)conv_input_.data(), (void*)input_nchw.deviceId(), input_nchw.size());

    auto conv_output = conv2d(conv_input_, conv_weight_, conv_bias_, inputs[0]->batch(), inputs[0]->height(), inputs[0]->width(), conv_common_param_);
    int batch = inputs[0]->batch();
    int channel = outputs[0]->channel();
    int height = outputs[0]->height();
    int width = outputs[0]->width();
    const int C4 = ((channel + 3) / 4) * 4;
    const int nc4hw4TotalSize = batch * C4 * height * width;
    std::vector<float> conv_output_nc4hwc4(nc4hw4TotalSize);
    MNN_ASSERT(outputs[0]->elementSize() == nc4hw4TotalSize);
    int areaOffset[2] = {height * width, height*width};
    MNNPackC4Common<float>(conv_output_nc4hwc4.data(), conv_output.data(),
                             height * width, channel, areaOffset);

    memcpy((void*)outputs[0]->deviceId(), (void*)conv_output_nc4hwc4.data(), outputs[0]->size());
    // for(int i = 0; i < nc4hw4TotalSize; ++i) {
    //     MNN_ASSERT(abs(conv_output_nc4hwc4[i] - ((float*)outputs[0]->deviceId())[i]) < 1e-3);
    //     if(abs(conv_output_nc4hwc4[i] - ((float*)outputs[0]->deviceId())[i]) > 1e-3) {
    //         MNN_PRINT("pack error at %d, %f, %f\n", i, conv_output_nc4hwc4[i], ((float*)outputs[0]->deviceId())[i]);
    //     } // "/model.20/cv3/conv/Conv_output_0"   "/model.24/m.1/Conv_output_0" "/model.21/conv/Conv_output_0" "/model.23/cv3/conv/Conv_output_0" "/model.24/m.2/Conv_output_0"
    // }
    return code;
}
ErrorCode XPUDenseConvolutionGeneralExecutor::onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) {
    mInputs = {inputs[0], mResource->mWeight.get(), mResource->mBias.get()};
    auto code = mProxy->onResize(mInputs, outputs);
    if (NO_ERROR != code) {
        return code;
    }
    return NO_ERROR;
}
bool XPUDenseConvolutionGeneralExecutor::initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<XPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes) {
    int weightLength = hU * lU * hP * lP;
    resource->mDequantize.bits = 8;
    resource->lU = lU;
    resource->hU = hU;
    resource->lP = lP;
    resource->hP = hP;
    MNN_ASSERT(lP == 1);
    // Save scale bias
    int dequantCnt = int8Info->alpha.size();
    int scaleSize = dequantCnt; // real size
    if (int8Info->asymmetric) {
        scaleSize = dequantCnt / 2;

    }
    int blockNum = scaleSize / outputCount;
    scaleSize = blockNum * hU * hP; // pack size
    resource->mDequantize.mScaleBias.reset(MNN::Tensor::createDevice<uint8_t>({scaleSize * 2 * bytes}));
    auto res = resource->backend->onAcquireBuffer(resource->mDequantize.mScaleBias.get(), Backend::STATIC);
    if (!res) {
        return false;
    }
    int originOffset = 0;
    auto srcWInt8 = int8Info->weight.get();
    std::vector<int8_t> blob;
    if (int8Info->canUseInt4) {
        // Revert int4 to int8
        auto size = int8Info->weight.size();
        blob.resize(int8Info->weight.size() * 2);
        auto idxBuf = (uint8_t*)srcWInt8;
        for (int i=0; i<size; ++i) {
            int val = idxBuf[i];
            int x1 = val / 16;
            int x2 = val % 16;
            blob[2 * i] = x1 - 8;
            blob[2 * i + 1] = x2 - 8;

        }
        srcWInt8 = blob.data();
    }
    {
        resource->mWeight.reset(Tensor::createDevice<int8_t>(std::vector<int>{hU, lU * lP, hP}));
        auto res = resource->backend->onAcquireBuffer(resource->mWeight.get(), Backend::STATIC);
        if (!res) {
            return false;
        }
        // Reorder weight for int8
        auto dstWInt8 = resource->mWeight->host<int8_t>();
        ::memset(dstWInt8, 0, resource->mWeight->usize());
        for (int y=0; y<outputCount; ++y) {
            int yo = y / hP;
            int yi = y % hP;
            auto srcY = srcWInt8 + y * srcChannel * kernelSize;
            auto dstY = dstWInt8 + yo * lP * hP * lU + yi;
            for (int iz=0; iz<srcChannel; ++iz) {
                for (int k=0; k<kernelSize; ++k) {
                    int sx = iz * kernelSize + k;
                    int dx = iz + k * srcChannel;
                    dstY[dx * hP] = srcY[sx];
                }
            }
        }
    }
    auto alphaPtr = resource->mDequantize.mScaleBias->host<float>();
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * bytes);
    ::memset(alphaPtr, 0, 2 * scaleSize * bytes);
    int h = int8Info->alpha.size();
    if (bytes == 2) {
        auto core = static_cast<XPUBackend*>(resource->backend)->functions();
        std::vector<float> tmpAlpha(scaleSize * 2, 0.0f);
        if (int8Info->asymmetric) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = tmpAlpha.data() + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[2 * scaleIndex + 1];
                    dstAlpha[j + scaleSize] = srcAlpha[2 * scaleIndex] + (float)originOffset * dstAlpha[j];
                }
            }
        } else {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = tmpAlpha.data() + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[scaleIndex];
                    dstAlpha[j + scaleSize] = (float)originOffset * dstAlpha[j];
                }
            }
        }
        core->MNNFp32ToLowp(tmpAlpha.data(), reinterpret_cast<int16_t*>(alphaPtr), scaleSize * 2);
    } else {
        if (int8Info->asymmetric) {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = alphaPtr + i * hU * hP;
                auto dstBias  = biasPtr + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[2 * scaleIndex + 1];
                    dstBias[j] = srcAlpha[2 * scaleIndex] + (float)originOffset * dstAlpha[j];
                }
            }
        } else {
            for (int i = 0; i < blockNum; ++i) {
                auto dstAlpha = alphaPtr + i * hU * hP;
                auto dstBias  = biasPtr + i * hU * hP;
                auto srcAlpha = int8Info->alpha.get();
                for (int j = 0; j < outputCount; ++j) {
                    int scaleIndex = j * blockNum + i;
                    dstAlpha[j] = srcAlpha[scaleIndex];
                    dstBias[j] = (float)originOffset * dstAlpha[j];
                }
            }
        }
    }
    return true;
}

void XPUDenseConvolutionGeneralExecutor::selectLowMemoryMatmulFunc(lowMemoryMatmulUnit* matmulUnit, lowMemoryMatmulRemain* matmulRemain, float* weightBytes, int32_t weightQuantBits, const XPUCoreFunctions* core) {
    if (weightQuantBits == 8) {
        *matmulUnit = core->MNNPackedMatMul_int8;
        *matmulRemain = core->MNNPackedMatMulRemain_int8;
        *weightBytes  = 1;
    }
}
void XPUDenseConvolutionGeneralExecutor::initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const XPUCoreFunctions* function) {
    XPUConvolutionTiledExecutor::initWeight(source, cache, depth, outputCount, kernelSize, function);
    function->MNNPackForMatMul_B(dest, cache, outputCount, kernelSize * depth, true);

}

ErrorCode ConvolutionTiledExecutorMultiInput::onExecute(const std::vector<Tensor*>& inputs,
                                                        const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = inputs[1]->batch();
    auto function = static_cast<XPUBackend*>(backend())->functions();
    if (nullptr != mTempBias) {
        ::memset(mTempBias->host<float>(), 0, mTempBias->elementSize() * function->bytes);
        if (inputs.size() > 2) {
            ::memcpy(mTempBias->host<float>(), inputs[2]->host<float>(), inputs[2]->elementSize() * function->bytes);
        }
    }
    auto cache = mTempWeightCache->host<float>();
    auto source = inputs[1]->host<float>();
    auto kernelSize = inputs[1]->stride(1);
    // Swap k, ic
    int dims[4] = {
        depth,
        kernelSize,
        kernelSize,
        depth
    };
    if (function->bytes < 4) {
        // TODO: Opt it
        // Lowp
        source = mTempWeightCache->host<float>() + mTempWeightCache->stride(0);
        function->MNNLowpToFp32(inputs[1]->host<int16_t>(), source, inputs[1]->elementSize());
        for (int o=0; o<outputCount; ++o) {
            auto dO = cache + o * depth * kernelSize;
            auto sO = source + o * depth * kernelSize;
            MNNTranspose32Bit((int32_t*)dO, (const int32_t*)sO, &dims[0]);
        }
        function->MNNFp32ToLowp(cache, (int16_t*)cache, inputs[1]->elementSize());
    } else {
        for (int o=0; o<outputCount; ++o) {
            auto dO = cache + o * depth * kernelSize;
            auto sO = source + o * depth * kernelSize;
            MNNTranspose32Bit((int32_t*)dO, (const int32_t*)sO, &dims[0]);
        }
    }
    function->MNNPackForMatMul_B(mTempWeight->host<float>(), mTempWeightCache->host<float>(), outputCount, kernelSize * depth, true);
    return mProxy->onExecute(mInputs, outputs);
}
ErrorCode ConvolutionTiledExecutorMultiInput::onResize(const std::vector<Tensor*>& inputs,
                                                       const std::vector<Tensor*>& outputs) {
    int depth       = inputs[1]->channel();
    int outputCount = outputs[0]->channel();
    auto function = static_cast<XPUBackend*>(backend())->functions();
    int eP, lP, hP;
    function->MNNGetMatMulPackMode(&eP, &lP, &hP);
    auto kernelSize = depth * inputs[1]->stride(1);
    mTempWeight.reset(Tensor::createDevice<float>(
        {UP_DIV(outputCount, hP), UP_DIV(kernelSize, lP), lP * hP}));
    if (function->bytes < 4) {
        mTempWeightCache.reset(Tensor::createDevice<int32_t>({2, outputCount * kernelSize}));
    } else {
        mTempWeightCache.reset(Tensor::createDevice<float>({outputCount * kernelSize}));
    }
    auto res = backend()->onAcquireBuffer(mTempWeight.get(), Backend::DYNAMIC);
    res = res && backend()->onAcquireBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    mTempBias.reset();
    if (!res) {
        return OUT_OF_MEMORY;
    }
    if (inputs.size() > 2 && inputs[2]->elementSize() % function->pack == 0) {
        mInputs = {inputs[0], mTempWeight.get(), inputs[2]};
    } else {
        mTempBias.reset(Tensor::createDevice<float>({UP_DIV(outputCount, function->pack) * function->pack}));
        backend()->onAcquireBuffer(mTempBias.get(), Backend::DYNAMIC);
        mInputs = {inputs[0], mTempWeight.get(), mTempBias.get()};
    }
    backend()->onReleaseBuffer(mTempWeightCache.get(), Backend::DYNAMIC);
    auto errorCode = mProxy->onResize(mInputs, outputs);
    backend()->onReleaseBuffer(mTempWeight.get(), Backend::DYNAMIC);
    if (nullptr != mTempBias) {
        backend()->onReleaseBuffer(mTempBias.get(), Backend::DYNAMIC);
    }
    return errorCode;
}


void DenseConvolutionTiledImpl::getPackParameter(int* eP, int* lP, int* hP, const XPUCoreFunctions* core) {
    core->MNNGetMatMulPackMode(eP, lP, hP);
    return;
}


XPUPerfConfig DenseConvolutionTiledImpl::bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                                                const Tensor *outputTensor, int threadNumber, Backend* b) {
    auto input   = inputTensor;
    Tensor *bias = nullptr;
    auto core    = static_cast<XPUBackend *>(b)->functions();
    int bytes    = core->bytes;
    int unit     = core->pack;
    int ePMax, lP, hP;
    core->MNNGetMatMulPackMode(&ePMax, &lP, &hP);
    auto kernel_width      = common->kernelX();
    auto kernel_height     = common->kernelY();
    auto output      = outputTensor;
    auto batch       = output->batch();
    auto width       = output->width();
    auto height      = output->height();
    auto src_width                = input->width();
    auto icC4                     = UP_DIV(input->channel(), unit);
    auto ic                       = input->channel();
    auto L                        = ic * common->kernelY() * common->kernelX();

    auto outputChannel = output->channel();
    auto padX = ConvolutionCommon::convolutionPad(inputTensor, outputTensor, common).first;
    if (src_width == 1 && width == 1 && height > 1 && kernel_width == 1 && padX == 0) {
        /* Swap x, y*/
        width         = height;
        height        = 1;
        kernel_width  = kernel_height;
        kernel_height = 1;
    }
    auto kernelSize               = common->kernelX() * common->kernelY();
    auto plane    = width * height * batch;
    auto oC4           = UP_DIV(outputChannel, unit);

     //In next major version these would be read from microbenchmark result file.
     constexpr int roofLine = 20;
     constexpr int indexCalculate = 3000;
     constexpr int indexMem = 40;

    XPUPerfConfig denseConfig(false, 0, 0, 0, std::numeric_limits<float>().max());

    for (int eP = ePMax; eP >= ePMax; eP -= 16) { // search space should be open after pack-free dense is available.
        int tileCount = UP_DIV(plane, eP);
        auto hTileCount = UP_DIV(outputChannel, hP);

        float outerFlops[3], innerFlops[3], outerBandwidth[3], innerBandwidth[3], outer[3], inner[3], outerAcc = 0, innerAcc = 0;
        float tailCost = 0.0, lastTail = 0.0;

        if (plane % eP == 0) {
            tailCost = 1.0f;
            lastTail = 1.0f;
        } else {
            bool moreThanOnetail = tileCount % threadNumber > 1;
            lastTail = (4.f * (plane % eP)) / eP;
            tailCost = moreThanOnetail ? (std::max(1.0f, lastTail)) : lastTail;
        }

        float outerCoefficient = tailCost + ((tileCount - 1) / threadNumber);
        float innerCoefficient = lastTail + ((plane - 1) / eP);

        int indexNumber = UP_DIV(eP, width) * kernel_width * kernel_height;
        outerFlops[0] = outerCoefficient * indexNumber * indexCalculate * unit;
        outerFlops[1] = 0;
        outerFlops[2] = outerCoefficient * eP * (2 * L) * oC4 * unit;
        outerBandwidth[0] = outerCoefficient * indexNumber * indexMem;
        outerBandwidth[1] = outerCoefficient * indexNumber * (2 * eP * ic);
        outerBandwidth[2] = outerCoefficient * (eP * 2 * L + oC4 * unit * 2 *  L + eP * oC4 * unit);

        innerFlops[0] = innerCoefficient * indexNumber * indexCalculate * unit;
        innerFlops[1] = 0;
        innerFlops[2] = innerCoefficient * eP * (2 * L) * UP_DIV(oC4, threadNumber) * unit;
        innerBandwidth[0] = innerCoefficient * indexNumber * indexMem;
        innerBandwidth[1] = innerCoefficient * (2 * eP * unit + 10 * sizeof(int) * unit) * UP_DIV(icC4 * indexNumber, threadNumber);
        innerBandwidth[2] = innerCoefficient * (eP * 2 * L + unit * 2*  L + eP * unit) * UP_DIV(oC4, threadNumber);

        for (int i = 0; i < sizeof(outerFlops) / sizeof(float); i++) {
             outer[i] = std::max(outerBandwidth[i] * roofLine, outerFlops[i]);
             inner[i] = std::max(innerBandwidth[i] * roofLine, innerFlops[i]);
             outerAcc += outer[i];
             innerAcc += inner[i];
        }
        XPUPerfConfig thisConfig(false, eP, eP, 0,  -1);
        thisConfig.isParallelInner = outerAcc > innerAcc && 0 == core->matmulBytes;
        thisConfig.instructionCosts = outerAcc > innerAcc ? innerAcc : outerAcc;

        if (thisConfig.instructionCosts < denseConfig.instructionCosts) {
            denseConfig = thisConfig;
        }
    }

    return denseConfig;

}

ErrorCode DenseConvolutionTiledImpl::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    XPUConvolution::onResize(inputs, outputs);
    auto input   = inputs[0];
    auto weight  = inputs[1];
    Tensor *bias = nullptr;
    if (inputs.size() > 2) {
        bias = inputs[2];
    }
    auto core    = static_cast<XPUBackend *>(backend())->functions();
    int bytes    = core->bytes;
    float weightBytes  = bytes;
    int unit     = core->pack;
    int matmulBytes = bytes;
    if (core->matmulBytes != 0) {
        matmulBytes = core->matmulBytes;
    }
    auto packA   = core->MNNPackC4ForMatMul_A;
    int eP, lP, hP;
    getPackParameter(&eP, &lP, &hP, core);
    auto matmulUnit   = core->MNNPackedMatMul;
    auto matmulRemain = core->MNNPackedMatMulRemain;
    const uint8_t* dequantAlpha = nullptr;
    const uint8_t* dequantBias = nullptr;
    auto ic       = input->channel();
    auto icC4     = UP_DIV(ic, unit);
    auto L        = ic * mCommon->kernelY() * mCommon->kernelX();
    auto tileC    = std::max(unit, hP);
    int blockSize = L;
    int blockNum  = 1;
    float halfStride = 1;
    size_t weightStride = 0;
#ifdef MNN_LOW_MEMORY
    if (mResource && mResource->mDequantize.bits <= 8) {
        MNN_ASSERT(mResource->mDequantize.bits == 8);
        XPUDenseConvolutionTiledExecutor::selectLowMemoryMatmulFunc(&matmulUnit, &matmulRemain, &weightBytes, mResource->mDequantize.bits, core);
        int scaleSize = mResource->mDequantize.mScaleBias->size() / (2 * bytes);
        blockNum = scaleSize / (mResource->hU * mResource->hP);
        blockSize /= blockNum;
        dequantAlpha = mResource->mDequantize.mScaleBias->host<uint8_t>();
        dequantBias = dequantAlpha + scaleSize * bytes;
        weightStride = (L - blockSize) * hP;
    }
#endif
    auto kernel_width      = mCommon->kernelX();
    auto kernel_height     = mCommon->kernelY();
    auto output      = outputs[0];
    auto batch       = output->batch();
    int threadNumber = ((XPUBackend *)backend())->threadNumber();

    int  LRoundup = ROUND_UP(L, lP);
    int  LRoundupC4 = UP_DIV(LRoundup, unit);
    auto outputChannel = output->channel();
    auto oC4      = UP_DIV(outputChannel, tileC);
    auto ocUp4    = ROUND_UP(outputChannel, hP);
    auto kernelSize               = mCommon->kernelX() * mCommon->kernelY();

    XPUConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParameters, mCommon, input, output, mPadX, mPadY, core, nullptr);
    mTempBufferTranspose.buffer().type          = halide_type_of<uint8_t>();
    mTempBufferTranspose.buffer().dimensions    = 2;
    mTempBufferTranspose.buffer().dim[0].extent = threadNumber;
    mTempBufferTranspose.buffer().dim[1].extent = UP_DIV(L, lP) * lP * eP * matmulBytes;
    TensorUtils::setLinearLayout(&mTempBufferTranspose);
    auto plane    = mIm2ColParameters.ow * mIm2ColParameters.oh * batch;
    int tileCount = UP_DIV(plane, eP);
    mConvPerfconfig = bestTileConvolutionConfig(mCommon, input, output, threadNumber, backend());
    bool success = backend()->onAcquireBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    if (!success) {
        return OUT_OF_MEMORY;
    }

    // auto bufferAlloc   = static_cast<XPUBackend *>(backend())->getBufferAllocator();
    auto maxLine       = UP_DIV(eP, mIm2ColParameters.ow) + 1;
    // auto tempPtr = bufferAlloc->alloc(kernelSize * maxLine * threadNumber * (4 * sizeof(int32_t) + sizeof(float *)));
    // if (tempPtr.invalid()) {
    //     return OUT_OF_MEMORY;
    // }
    auto tempPtr = malloc(kernelSize * maxLine * threadNumber * (4 * sizeof(int32_t) + sizeof(float *)));
    // backend()->onReleaseBuffer(&mTempBufferTranspose, Backend::DYNAMIC);
    // bufferAlloc->free(tempPtr);

    auto postParameters    = getPostParameters();
    mFunction.first        = threadNumber;

    if (mConvPerfconfig.isParallelInner) {
        auto rt = static_cast<const XPURuntime*>(backend()->getRuntime());
        std::vector<int> ocC4ParralSize(threadNumber + 1);
        ocC4ParralSize[0] = 0;
        static_cast<XPUBackend *>(backend())->computeDivideSizes(oC4, ocC4ParralSize.data()+1);
        mFunction.second = [=](int placeholder) {
        const float* biasPtr = bias ? bias->host<float>() : nullptr;
        auto gemmBuffer = mTempBufferTranspose.host<uint8_t>() + mTempBufferTranspose.stride(0) * 0;
        auto srcPtr     = (float const **)((uint8_t*)tempPtr + 0 * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
        auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
        auto weightPtr = weight->host<uint8_t>();

        constexpr int InfoSize = 4;
        int32_t shapeInfo[InfoSize];
        int32_t* info = shapeInfo;
        info[1] = mIm2ColParameters.iw * mIm2ColParameters.ih * batch;
        info[2] = eP;
        info[3] = mIm2ColParameters.strideX;
        size_t shapeParameters[PARAMETERSIZE];
        size_t* parameters = shapeParameters;
        parameters[0]          = eP * bytes;
        parameters[1]          = blockSize;
        parameters[2]          = outputChannel;
        parameters[3]          = plane * unit * bytes;
        parameters[4]          = 0;
        parameters[5]          = weightStride; // Only used when block quant
        parameters[6]          = 0;

        auto dstOrigin = output->host<uint8_t>();
        auto srcOrigin = input->host<uint8_t>();
        std::vector<int> im2colParallelSize(threadNumber + 1);
        im2colParallelSize[0] = 0;

        for (int x = 0; x < tileCount; x += 1) {
            int start  = (int)x * eP;
            int remain = plane - start;
            int xC     = remain > eP ? eP : remain;
            auto res = XPUConvolutionTiledExecutor::turnIm2ColToBlitInfo(srcPtr, el, start, xC, mIm2ColParameters, srcOrigin, bytes);
            int number    = res.first;
            bool needZero = res.second;
            info[0] = number;
            if (needZero || lP != 1) {
                ::memset(gemmBuffer, 0, mTempBufferTranspose.stride(0));
            }
            info[0] = 1;
            int hw4Stride = info[1] * unit * bytes;
            static_cast<XPUBackend *>(backend())->computeDivideSizes(number * icC4, im2colParallelSize.data() + 1);
            im2colParallelSize[0] = 0;
            MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                int threadEL[4];
                int ticSta = im2colParallelSize[tId];
                int ticEnd = im2colParallelSize[tId+1];
                for(int tic_inumber = ticSta; tic_inumber < ticEnd; tic_inumber++) {
                        int inumber = tic_inumber / icC4;
                        int t_ic = tic_inumber % icC4;
                        memcpy(threadEL, el + 4 * inumber, 4 * sizeof(int));
                        threadEL[1] = std::min(ic - (t_ic * unit), unit);
                        const float* source = (const float*)((const uint8_t*)(srcPtr[inumber]) + t_ic * hw4Stride);
                        auto gemmDest = gemmBuffer + t_ic * unit * eP * bytes;
                        packA((float *)gemmDest, &source, info, threadEL);
                }
            }
            MNN_CONCURRENCY_END();

            if (xC == eP) {
                MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                    size_t paraParameters[PARAMETERSIZE];
                    memcpy(paraParameters, parameters, PARAMETERSIZE * sizeof(size_t));
                    for (int t_oc = ocC4ParralSize[tId]; t_oc < ocC4ParralSize[tId+1]; ++t_oc) {
                        int ocIndex = t_oc * tileC;
                        auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + (ocIndex / unit * plane + start) * unit * bytes);
                        auto _weightFloatPtr = reinterpret_cast<const float*>(weightPtr + int((ocIndex / hP * LRoundup * hP) * weightBytes));
                        auto _biasFloatPtr = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(biasPtr) + ocIndex * bytes);
                        paraParameters[2] = std::min(outputChannel - ocIndex, tileC);
                        auto k = reinterpret_cast<const uint8_t*>(dequantAlpha + ocIndex * bytes);
                        auto b = reinterpret_cast<const uint8_t*>(dequantBias + ocIndex * bytes);
                        const float* relufp32 = nullptr;
                        const float* exeBiasPtr = nullptr;
                        int finishedL = 0;
                        int wquantStride = 0;
                        auto _weightPtr = reinterpret_cast<const int8_t*>(_weightFloatPtr);
                        uint8_t*  _APtr      = reinterpret_cast<uint8_t*>(gemmBuffer);
                        for (int bk = 0; bk < blockNum; ++bk) {
                            paraParameters[6] = bk;
                            if (bk == blockNum - 1) {
                                relufp32 = postParameters.data();
                                exeBiasPtr = _biasFloatPtr;
                            }
                            finishedL = blockSize * bk;
                            wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);
                            matmulUnit(_dstFloatPtr, (float*)(_APtr + eP * finishedL * bytes), (float*)(_weightPtr + wquantStride), paraParameters, relufp32, exeBiasPtr, (float*)(k + bk * ocUp4 * bytes), (float*)(b + bk * ocUp4 * bytes));
                        }
                    }
                }
                MNN_CONCURRENCY_END();
            } else {
                MNN_CONCURRENCY_BEGIN(tId, threadNumber) {
                    size_t paraParameters[PARAMETERSIZE];
                    memcpy(paraParameters, parameters, PARAMETERSIZE * sizeof(size_t));
                    for (int t_oc = ocC4ParralSize[tId]; t_oc < ocC4ParralSize[tId+1]; ++t_oc) {
                        int ocIndex = t_oc * tileC;
                        auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + (ocIndex / unit * plane + start) * unit * bytes);
                        auto _weightFloatPtr = reinterpret_cast<const float*>(weightPtr + int((ocIndex / hP * LRoundup * hP) * weightBytes));
                        auto _biasFloatPtr = reinterpret_cast<const float*>(reinterpret_cast<const uint8_t*>(biasPtr) + ocIndex * bytes);
                        paraParameters[2] = std::min(outputChannel - ocIndex, tileC);
                        auto k = reinterpret_cast<const uint8_t*>(dequantAlpha + ocIndex * bytes);
                        auto b = reinterpret_cast<const uint8_t*>(dequantBias + ocIndex * bytes);
                        const float* relufp32 = nullptr;
                        const float* exeBiasPtr = nullptr;
                        int finishedL = 0;
                        int wquantStride = 0;
                        const int8_t* _weightPtr = reinterpret_cast<const int8_t*>(_weightFloatPtr);
                        uint8_t*  _APtr      = reinterpret_cast<uint8_t*>(gemmBuffer);
                        for (int bk = 0; bk < blockNum; ++bk) {
                            paraParameters[6] = bk;
                            if (bk == blockNum - 1) {
                                relufp32 = postParameters.data();
                                exeBiasPtr = _biasFloatPtr;
                            }
                            finishedL = blockSize * bk;
                            wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);
                            matmulRemain(_dstFloatPtr, (float*)(_APtr + eP * finishedL * bytes), (float*)(_weightPtr + wquantStride), xC, paraParameters, relufp32, exeBiasPtr, (float*)(k + bk * ocUp4 * bytes), (float*)(b + bk * ocUp4 * bytes));
                        }
                    }
                }
                MNN_CONCURRENCY_END();
            }

        }
    };

    } else {
        std::vector<int> divides(threadNumber + 1);
        divides[0] = 0;

        static_cast<XPUBackend *>(backend())->computeDivideSizes(tileCount, divides.data() + 1);

        mFunction.second       = [=](int tId) {
            const float* biasPtr = bias ? (float*)bias->deviceId() : nullptr;
            auto gemmBuffer = (uint8_t*)mTempBufferTranspose.deviceId() + mTempBufferTranspose.stride(0) * tId;
            auto srcPtr     = (float const **)((uint8_t*)tempPtr + tId * kernelSize * maxLine * (4 * sizeof(int32_t) + sizeof(float *)));
            auto el         = (int32_t *)(srcPtr + kernelSize * maxLine);
            auto weightPtr = (float*)weight->deviceId();
            int32_t info[4];
            info[1] = mIm2ColParameters.iw * mIm2ColParameters.ih * batch;
            info[2] = eP;
            info[3] = mIm2ColParameters.strideX;
            size_t parameters[PARAMETERSIZE];
            parameters[0]          = eP * bytes;
            parameters[1]          = blockSize;
            parameters[2]          = outputChannel;
            parameters[3]          = plane * unit * bytes;
            parameters[4]          = 0;
            parameters[5]          = weightStride; // Only used when block quant
            parameters[6]          = 0;

            auto dstOrigin = (uint8_t*)output->deviceId();
            auto srcOrigin = (uint8_t*)input->deviceId();
            int tEnd = divides[tId+1];
            int tStart = divides[tId];
            for (int x = (int)tStart; x < tEnd; ++x) {
                int start  = (int)x * eP;
                int remain = plane - start;
                int xC     = remain > eP ? eP : remain;
                auto res = XPUConvolutionTiledExecutor::turnIm2ColToBlitInfo(srcPtr, el, start, xC, mIm2ColParameters, srcOrigin, bytes);
                auto number = res.first;
                bool needZero = res.second;
                info[0] = number;
                if (needZero || lP != 1) {
                    ::memset(gemmBuffer, 0, mTempBufferTranspose.stride(0));
                }

                if (number > 0) {
                    packA((float *)gemmBuffer, srcPtr, info, el);
                }

                int finishedL = 0;
                int wquantStride = 0;
                int8_t* _weightPtr = reinterpret_cast<int8_t*>(weightPtr);
                auto _dstFloatPtr = reinterpret_cast<float*>(dstOrigin + start * unit * bytes);
                const float* relufp32 = nullptr;
                const float* exeBiasPtr = nullptr;
                if (xC == eP) {
                    // matmulUnit(_dstFloatPtr, (float*)gemmBuffer, (float*)weightPtr, parameters, postParameters.data(), biasPtr, k, b);
                    for (int bk = 0; bk < blockNum; ++bk) {
                        parameters[6] = bk;
                        if (bk == blockNum - 1) {
                            relufp32 = postParameters.data();
                            exeBiasPtr = biasPtr;
                        }
                        finishedL = blockSize * bk;
                        wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);

                        matmulUnit(_dstFloatPtr, (float*)(gemmBuffer + bytes * eP * finishedL), (float*)(_weightPtr + wquantStride), parameters, relufp32, exeBiasPtr, (float*)(dequantAlpha + bk * ocUp4 * bytes), (float*)(dequantBias + bk * ocUp4 * bytes));
                    }
                } else {
                    for (int bk = 0; bk < blockNum; ++bk) {
                        parameters[6] = bk;
                        if (bk == blockNum - 1) {
                            relufp32 = postParameters.data();
                            exeBiasPtr = biasPtr;
                        }
                        finishedL = blockSize * bk;
                        wquantStride = static_cast<int32_t>(blockSize * bk * hP * halfStride);

                        matmulRemain(_dstFloatPtr, (float*)(gemmBuffer + eP * bytes * finishedL), (float*)(_weightPtr + wquantStride), xC, parameters, relufp32, exeBiasPtr, (float*)(dequantAlpha + bk * ocUp4 * bytes), (float*)(dequantBias + bk * ocUp4 * bytes ));
                    }
                    // matmulRemain(_dstFloatPtr, (float*)gemmBuffer, (float*)weightPtr, xC, parameters, postParameters.data(), biasPtr, k, b);
                }
            }
        };
    }
    return NO_ERROR;
}

ErrorCode DenseConvolutionTiledImpl::onExecute(const std::vector<Tensor*>& inputs,
                                          const std::vector<Tensor*>& outputs) {
    mFunction.second(0);
    // if (mConvPerfconfig.isParallelInner) {
    //     mFunction.second(0);
    // } else {
    //     MNN_CONCURRENCY_BEGIN(tId, mFunction.first) {
    //         mFunction.second((int)tId);
    //     }
    //     MNN_CONCURRENCY_END();
    // }

    return NO_ERROR;
}

} // namespace XPU
} // namespace MNN
