//
//  XPUConvolutionTiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef XPU_CONVOLUTION_TILED_EXECUTOR_HPP
#define XPU_CONVOLUTION_TILED_EXECUTOR_HPP


#include <functional>
#include "backend/xpu/execution/XPUConvolution.hpp"

// Tiled Slide Window or Im2Col + GEMM
namespace MNN {
namespace XPU {
class ConvolutionTiledImpl : public XPUConvolution {
public:
    ConvolutionTiledImpl(const Convolution2DCommon *common, Backend *b) : XPUConvolution(common, b) {
        // Do nothing
    }
    virtual ~ConvolutionTiledImpl() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual void getPackParameter(int* eP, int* lP, int* hP, const XPUCoreFunctions* core) = 0;

protected:
    Tensor mTempBufferTranspose;
    ConvolutionCommon::Im2ColParameter mIm2ColParameters;
    std::pair<int, std::function<void(int)>> mFunction;
    const XPUConvolution::Resource* mResource = nullptr;
};

class XPUConvolutionTiledExecutor : public Execution {
public:
    XPUConvolutionTiledExecutor(Backend* b, const float* bias, size_t biasSize);
    XPUConvolutionTiledExecutor(std::shared_ptr<XPUConvolution::Resource> res, Backend* b);
    virtual ~XPUConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_EXECUTION;
    }
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override {
        return NO_EXECUTION;
    }
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void initWeight(const float *source, float* cache, int depth, int outputCount, int kernelSize, const XPUCoreFunctions* function);
    static std::pair<int, bool> turnIm2ColToBlitInfo(float const ** srcPtr, int32_t* el, int start, int xC, const ConvolutionCommon::Im2ColParameter& im2Col, const uint8_t* srcOrigin, int bytes);
    static void setIm2ColParameter(ConvolutionCommon::Im2ColParameter& dstIm2ColParamter, const Convolution2DCommon* convCommon, Tensor* input, Tensor* output, int padX, int padY, const XPUCoreFunctions* floatCore, const CoreInt8Functions* int8Core, int pack = 0, int32_t* int8GemmUnit = nullptr);
    // Total / Stride
    static std::pair<size_t, std::pair<size_t, size_t>> computeBlitInfoSize(int eP, int ow, int kernelSize, int threadNumber);

protected:
    std::vector<Tensor *> mInputs;
    std::shared_ptr<XPUConvolution::Resource> mResource;
};

} // namespace XPU
} // namespace MNN

#endif // XPU_CONVOLUTION_TILED_EXECUTOR_HPP
