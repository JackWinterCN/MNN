//
//  ConvInt8TiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef CONV_INT8_TILED_EXECUTOR
#define CONV_INT8_TILED_EXECUTOR

#include "backend/xpu/execution/XPUConvolution.hpp"
#include "XPUInt8FunctionsOpt.hpp"
#include "XPUCommonOptFunction.hpp"

namespace MNN {
namespace XPU {

typedef void (*weightSummerFuncion)(float* kernlesum, int8_t* source, size_t outside, size_t reduceAxis, size_t hP, size_t lP);
class ConvInt8TiledExecutor : public XPUConvolution {
public:
    // given weight+bias+scale, do post process
    ConvInt8TiledExecutor(Backend* backend, const Op* op);
    ConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ResourceInt8> res);
    virtual ~ConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    virtual void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const XPUCoreInt8Functions* core) = 0;
    static void packWeightAndQuantInfo(int8_t* dstbuffer, const int8_t* weight, const int8_t* quantInfo, int32_t* info, int infoBytes = 4);
    static void reorderWeight(uint8_t* dst, const uint8_t* src, int32_t* info, int32_t initval = 0, float* kernelsum = nullptr, weightSummerFuncion summerFunc = nullptr);
    static void initializeConvInt8QuantInfo(std::shared_ptr<XPUConvolution::ResourceInt8>& resourceInt8, const Convolution2D* conv2D);

protected:
    ConvolutionCommon::Im2ColParameter mIm2ColParamter;
    int mTileCount;
    int mThreadNums;
    std::shared_ptr<Tensor> mTempIm2ColBuffer;
    std::shared_ptr<XPUConvolution::ResourceInt8> mResourceInt8;
    std::shared_ptr<XPUConvolution::MutableResourceInt8> mMutableResource;
    // MemChunk mBlitInfo;
    void* mBlitInfo{nullptr};
    std::pair<size_t, size_t> mBlitInfoStride;
    int mIm2ColCount;
};

//
//  DenseConvInt8TiledExecutor.hpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//


class DenseConvInt8TiledExecutor : public ConvInt8TiledExecutor {
public:
    DenseConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant); // dynamic quant
    virtual ~DenseConvInt8TiledExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const XPUCoreInt8Functions* core) override;
private:
    DenseConvInt8TiledExecutor(Backend* backend, const Op* op, const DenseConvInt8TiledExecutor& exe);

    decltype(XPUCoreInt8Functions::Int8GemmKernel) mGemmKernel;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, ssize_t)> mQuantFunc;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, size_t, size_t)> mQuantAndReorderFunc = nullptr;
    std::function<void(float* dest, int8_t* source, const float* scale, ssize_t realDstCount, SumByAxisParams sumParams)> mSumByAxisLFunc;
    std::shared_ptr<Tensor> mQuantInput;
    std::shared_ptr<Tensor> mDynamicBias;
    std::shared_ptr<Tensor> mAccumBuffer;
    std::shared_ptr<Tensor> mBatchQuantInfo;
    // MemChunk mTempMaxMinValueBuffer;
    void* mTempMaxMinValueBuffer{nullptr};
    // MemChunk mTempSrcSum;
    void* mTempSrcSum{nullptr};
    // MemChunk mQScaleZero;
    void* mQScaleZero;
    // MemChunk mReorderBuffer;
    void* mReorderBuffer;
    // MemChunk mBiasBufferFusedInputzero;
    void* mBiasBufferFusedInputzero;
    std::vector<int32_t> mDivides;

    int mThreadNums;
    int mBlockNum = 1;
    int mInputBlockNum = 1;
    int mOcPerThread;
    bool mSplitByOc;
    bool mUseBatchQuan;
    bool mIm2ColBasedInt8;
    int mSizeInputBlockQuant;
    bool mToFuseInputbias2Bias;
#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI::AccelType mAccelType = KleidiAI::AccelType::ACC_TYPE_NUMBER;
#endif
};


class DenseConvInt8TiledGeneralExecutor : public ConvInt8TiledExecutor {
public:
    DenseConvInt8TiledGeneralExecutor(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant); // dynamic quant
    virtual ~DenseConvInt8TiledGeneralExecutor();
    virtual ErrorCode onResize(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual ErrorCode onExecute(const std::vector<Tensor *> &inputs, const std::vector<Tensor *> &outputs) override;
    virtual bool onClone(Backend* bn, const Op* op, Execution** dst) override;
    void getPackParameter(int* Unit, int* SrcUnit, int* DestUnit, const XPUCoreInt8Functions* core) override;
private:
    DenseConvInt8TiledGeneralExecutor(Backend* backend, const Op* op, const DenseConvInt8TiledGeneralExecutor& exe);

    decltype(XPUCoreInt8Functions::Int8GemmKernel) mGemmKernel;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, ssize_t)> mQuantFunc;
    std::function<void(const float*, int8_t*, size_t, const float*, ssize_t, ssize_t, const float*, size_t, size_t)> mQuantAndReorderFunc = nullptr;
    std::function<void(float* dest, int8_t* source, const float* scale, ssize_t realDstCount, SumByAxisParams sumParams)> mSumByAxisLFunc;
    std::shared_ptr<Tensor> mQuantInput;
    std::shared_ptr<Tensor> mDynamicBias;
    std::shared_ptr<Tensor> mAccumBuffer;
    std::shared_ptr<Tensor> mBatchQuantInfo;
    // MemChunk mTempMaxMinValueBuffer;
    void* mTempMaxMinValueBuffer{nullptr};
    // MemChunk mTempSrcSum;
    void* mTempSrcSum{nullptr};
    // MemChunk mQScaleZero;
    void* mQScaleZero;
    // MemChunk mReorderBuffer;
    void* mReorderBuffer;
    // MemChunk mBiasBufferFusedInputzero;
    void* mBiasBufferFusedInputzero;
    std::vector<int32_t> mDivides;

    int mThreadNums;
    int mBlockNum = 1;
    int mInputBlockNum = 1;
    int mOcPerThread;
    bool mSplitByOc;
    bool mUseBatchQuan;
    bool mIm2ColBasedInt8;
    int mSizeInputBlockQuant;
    bool mToFuseInputbias2Bias;
    Convolution2DCommonT conv_common_param_;
    std::vector<float> conv_bias_;
    std::vector<float> conv_weight_;
    std::vector<float> conv_input_;
    std::vector<int8_t> init_weight_;
    std::vector<float> init_alpha_;
    std::vector<float> out_scales_;
    float scale_in_;
#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI::AccelType mAccelType = KleidiAI::AccelType::ACC_TYPE_NUMBER;
#endif
};


} // namespace XPU
} // namespace MNN

#endif /* CONV_INT8_TILED_EXECUTOR */
