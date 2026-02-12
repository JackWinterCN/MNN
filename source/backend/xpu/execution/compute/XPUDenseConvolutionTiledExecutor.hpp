//
//  XPUDenseConvolutionTiledExecutor
//  MNN
//
//  Created by MNN on 2018/07/16.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef XPU_DENSE_CONVOLUTION_TILED_EXECUTOR_HPP
#define XPU_DENSE_CONVOLUTION_TILED_EXECUTOR_HPP


#include <functional>
#include <vector>
#include "backend/xpu/execution/XPUConvolution.hpp"
#include "XPUConvolutionTiledExecutor.hpp"
// Tiled Slide Window or Im2Col + GEMM
namespace MNN {
namespace XPU {
typedef void(*lowMemoryMatmulUnit)(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
typedef void(*lowMemoryMatmulRemain)(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b);
class DenseConvolutionTiledImpl : public ConvolutionTiledImpl {
public:
    DenseConvolutionTiledImpl(const Convolution2DCommon *common, Backend *b, XPUConvolution::Resource* resource = nullptr) : ConvolutionTiledImpl(common, b) {
        mResource = resource;
    }
    ErrorCode onResize(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    ErrorCode onExecute(const std::vector<Tensor*>& inputs,
                                         const std::vector<Tensor*>& outputs) override;
    virtual ~DenseConvolutionTiledImpl() = default;
    void getPackParameter(int* eP, int* lP, int* hP, const XPUCoreFunctions* core) override;
    static XPUPerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b);
protected:
};
class XPUDenseConvolutionTiledExecutor : public XPUConvolutionTiledExecutor {
public:
    XPUDenseConvolutionTiledExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                  size_t originWeightSize, const float *bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common>);

    XPUDenseConvolutionTiledExecutor(std::shared_ptr<XPUConvolution::Resource> res, const Convolution2DCommon *common, Backend* b);
    virtual ~XPUDenseConvolutionTiledExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const XPUCoreFunctions* function);
    static XPUPerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
        return DenseConvolutionTiledImpl::bestTileConvolutionConfig(common, inputTensor, outputTensor, threadNumber, b);
    }
    static bool initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<XPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes);
    static void selectLowMemoryMatmulFunc(lowMemoryMatmulUnit* matmulUnit, lowMemoryMatmulRemain* matmulRemain, float* weightBytes, int32_t weightQuantBits, const XPUCoreFunctions* core);
    struct DequantizeCache {
        std::shared_ptr<MNN::Tensor> weight;
        std::shared_ptr<MNN::Tensor> weightInt8;
    };
protected:
    DequantizeCache mWeightCache;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
};


class XPUDenseConvolutionGeneralExecutor : public XPUConvolutionTiledExecutor {
public:
    XPUDenseConvolutionGeneralExecutor(const Convolution2DCommon *common, Backend *b, const float *originWeight,
                                  size_t originWeightSize, const float *bias, size_t biasSize, std::shared_ptr<ConvolutionCommon::Int8Common>);

    XPUDenseConvolutionGeneralExecutor(std::shared_ptr<XPUConvolution::Resource> res, const Convolution2DCommon *common, Backend* b);
    virtual ~XPUDenseConvolutionGeneralExecutor();

    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void initWeight(float *dest, const float *source, float* cache, int depth, int outputCount, int kernelSize, const XPUCoreFunctions* function);
    static XPUPerfConfig bestTileConvolutionConfig(const Convolution2DCommon *common, const Tensor *inputTensor,
                                          const Tensor *outputTensor, int threadNumber, Backend* b) {
        return DenseConvolutionTiledImpl::bestTileConvolutionConfig(common, inputTensor, outputTensor, threadNumber, b);
    }
    static bool initQuantizeResource(std::shared_ptr<ConvolutionCommon::Int8Common> int8Info, std::shared_ptr<XPUConvolution::Resource> resource, int hU, int hP, int lU, int lP, int outputCount, int srcChannel, int kernelSize, int bytes);
    static void selectLowMemoryMatmulFunc(lowMemoryMatmulUnit* matmulUnit, lowMemoryMatmulRemain* matmulRemain, float* weightBytes, int32_t weightQuantBits, const XPUCoreFunctions* core);
    struct DequantizeCache {
        std::shared_ptr<MNN::Tensor> weight;
        std::shared_ptr<MNN::Tensor> weightInt8;
    };
protected:
    DequantizeCache mWeightCache;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
    Convolution2DCommonT conv_common_param_;
    std::vector<float> conv_bias_;
    std::vector<float> conv_weight_;
    std::vector<float> conv_input_;
};

class ConvolutionTiledExecutorMultiInput : public Execution {
public:
    ConvolutionTiledExecutorMultiInput(const Convolution2DCommon *common, Backend *b) : Execution(b) {
        mProxy.reset(new DenseConvolutionTiledImpl(common, b));
    }
    virtual ~ConvolutionTiledExecutorMultiInput() = default;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;

private:
    std::shared_ptr<Tensor> mTempWeight;
    std::shared_ptr<Tensor> mTempWeightCache;
    std::shared_ptr<Tensor> mTempBias;
    std::shared_ptr<DenseConvolutionTiledImpl> mProxy;
    std::vector<Tensor *> mInputs;
};
} // namespace XPU
} // namespace MNN

#endif // XPU_DENSE_CONVOLUTION_TILED_EXECUTOR_HPP
