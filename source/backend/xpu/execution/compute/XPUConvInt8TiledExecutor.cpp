//  ConvInt8TiledExecutor.cpp
//  MNN
//
//  Created by MNN on 2019/5/17.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "XPUConvInt8TiledExecutor.hpp"
#include "XPUConvolutionTiledExecutor.hpp"
#include "core/Macro.h"
#include "core/BufferAllocator.hpp"

#include <math.h>
#include "backend/xpu/backend/XPUBackend.hpp"
#include "backend/xpu/execution/compute/Concurrency.h"
#include "backend/xpu/execution/XPUTensorConvert.hpp"
#include "core/TensorUtils.hpp"

#define QUANT_INFO_BYTES 4
namespace MNN {
namespace XPU {

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Op* op): XPUConvolution(op->main_as_Convolution2D()->common(), backend) {}

ConvInt8TiledExecutor::ConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ResourceInt8> res): XPUConvolution(op->main_as_Convolution2D()->common(), backend), mResourceInt8(res) {
    if (!res->mDynamicQuant) {
        mMutableResource.reset(new MutableResourceInt8(res, backend));
        mValid = mMutableResource->mValid;
    }
}

ConvInt8TiledExecutor::~ConvInt8TiledExecutor() {
    // Do nothing
}

bool ConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    return false;
}

ErrorCode ConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    if (nullptr != mMutableResource) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
    }
    XPUConvolution::onResize(inputs, outputs);
    XPUConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, static_cast<XPUBackend*>(backend())->functions(), static_cast<XPUBackend*>(backend())->int8Functions());
    return NO_ERROR;
}

void ConvInt8TiledExecutor::initializeConvInt8QuantInfo(std::shared_ptr<XPUConvolution::ResourceInt8> &resourceInt8, const Convolution2D *conv2D) {
    // input/output scale&zeorpoint
    resourceInt8->mActBits = 8;
    resourceInt8->mWeightAsymmetricQuant = false;
    if (conv2D->symmetricQuan()) {
        resourceInt8->mActBits = conv2D->symmetricQuan()->nbits();
    }
    if (conv2D->bias() && conv2D->quanParameter()->alpha()) {
        resourceInt8->mUseConvQuan = false;
    }
    resourceInt8->mInputZeroPoint = 0;
    resourceInt8->mOutputZeroPoint = 0;
    resourceInt8->mClampMin = -128;
    resourceInt8->mClampMax = 127;
    if (conv2D->symmetricQuan()) {
        resourceInt8->mInputZeroPoint = conv2D->symmetricQuan()->zeroPoint();
        resourceInt8->mOutputZeroPoint = conv2D->symmetricQuan()->outputZeroPoint();
        resourceInt8->mClampMin = conv2D->symmetricQuan()->clampMin();
        resourceInt8->mClampMax = conv2D->symmetricQuan()->clampMax();
    }
    if (conv2D->quanParameter() != nullptr) {
        resourceInt8->mInputScale = conv2D->quanParameter()->scaleIn();
        resourceInt8->mOutputScale = conv2D->quanParameter()->scaleOut();
    }
    resourceInt8->mRelu = conv2D->common()->relu() || conv2D->common()->relu6();
    if (conv2D->symmetricQuan() && conv2D->symmetricQuan()->outputDataType() == MNN::DataType_DT_FLOAT) {
        resourceInt8->mOutputZeroPoint = 0;
        resourceInt8->mOutputScale = 1.0f;
    }
}

void ConvInt8TiledExecutor::reorderWeight(uint8_t* dst, const uint8_t* src, int32_t* info, int32_t initval, float* kernelsum, weightSummerFuncion summerFunc) {
    // weight shape = {UP_DIV(oc, UNIT), blockNum, kernelCount* UP_DIV(ic / blockNum, SRC_UNIT), UNIT, SRC_UNIT};
    MNN_ASSERT(dst != nullptr && src != nullptr);
    
    int blockNum = info[0];
    int oc = info[1];
    int ic = info[2];
    int kernelCount = info[3];
    int UNIT = info[4];
    int SRC_UNIT = info[5];

    int blockL  = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    int stride0 = blockNum * SRC_UNIT * blockL * UNIT;    // weight->stride(0)
    int stride1 = blockL * SRC_UNIT * UNIT;               // weight->stride(1)
    int stride2 = UNIT * SRC_UNIT;                        // weight->stride(2)
    int weightlen = stride0 * UP_DIV(oc, UNIT);
    memset(dst, initval, weightlen);

    auto hU = UP_DIV(oc, UNIT);
    auto lU = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    bool fast = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && ROUND_UP(ic, SRC_UNIT) == ic);
    if (fast) {
        for (int i = 0; i < hU; ++i) {
            for (int k = 0; k < UNIT; ++k) {
                for (int bl = 0; bl < blockNum; ++bl) {
                    for (int j = 0; j < blockL; ++j) {
                        int srcindex = (i * UNIT + k) * ic + bl * (lU * SRC_UNIT) + j * SRC_UNIT;
                        int dstindex = i * stride0 + bl * stride1 + j * stride2 + k * SRC_UNIT;
                        memcpy(dst + dstindex, src + srcindex, SRC_UNIT);
                    }
                }
            }
        }
    } else {
        AutoStorage<uint8_t> tmpBuffer(ic * kernelCount * ROUND_UP(oc, UNIT));
        memset(tmpBuffer.get(), 0, tmpBuffer.size());
        
        auto area = ic * kernelCount;
        // [oc, ic, k2] -> [hU, ic, k2, hP]
        for (int i = 0; i < oc; ++i) {
            auto outId = i / UNIT;
            auto inId  = i % UNIT;
            for (int j = 0; j < area; ++j) {
                tmpBuffer.get()[outId * area * UNIT + j * UNIT + inId] = src[i * area + j];
            }
        }
        // [hU, ic, (k2, hP)] -> [hU, blocknum, lU, (k2, hP), lP]
        AutoStorage<uint8_t> packedBuffer(weightlen);
        memset(packedBuffer.get(), 0, weightlen);
        area = kernelCount * UNIT;
        auto blockic = ic / blockNum;
        for (int i = 0; i < hU; ++i) {
            for (int j = 0; j < ic; ++j) {
                int bk = j / blockic;
                int blu = (j % blockic) / SRC_UNIT;
                int blp = (j % blockic) % SRC_UNIT;
                for (int k = 0; k < area; ++k) {
                    int dstindex = i * stride0 + bk * stride1 + blu * kernelCount * stride2 + k * SRC_UNIT + blp;
                    int srcindex = i * ic * area + j * area + k;
                    packedBuffer.get()[dstindex] = tmpBuffer.get()[srcindex];
                }
            }
        }
        // [(hU, blocknum), lU, k2, (hP, lP)] -> [(hU, blocknum), k2, lU, (hP, lP)]
        area = UNIT * SRC_UNIT;
        auto bklU = UP_DIV(ic, SRC_UNIT) / blockNum;
        for (int bk = 0; bk < blockNum * hU; ++bk) {
            for (int i = 0; i < kernelCount; ++i) {
                for (int j = 0; j < bklU; ++j) {
                    memcpy(dst + bk * stride1 + i * bklU * area + j * area, packedBuffer.get() + bk * stride1 + i * area + j * kernelCount * area, area);
                }
            }
        }
    } // not fast

    if (summerFunc != nullptr && kernelsum != nullptr) {
        summerFunc(kernelsum, (int8_t*)dst, blockNum * hU, blockL, UNIT, SRC_UNIT);
    }
}

void ConvInt8TiledExecutor::packWeightAndQuantInfo(int8_t* dstbuffer, const int8_t* weight, const int8_t* quantInfo, int32_t* info, int infoBytes) {
    int blockNum    = info[0];
    int ocDiv       = info[1];
    int blockL      = info[2];
    int UNIT        = info[3];
    int SRC_UNIT    = info[4];
    auto ocUp4      = info[5];
    auto src0 = weight;              // int8 weight: [oc/hp, blocknum, ic/lp*(kx*ky)/blocknum, hp, lp]
    auto src1 = quantInfo;           // dequant scale: [blocknum, ocUp4]
    auto src2 = src1 + infoBytes * ocUp4 * blockNum; // dequant bias: [blocknum, ocUp4]
    int stride0 = info[0] * info[2] * info[3] * info[4];
    int stride1 = info[2] * info[3] * info[4];
    
    // dst: [oc/hp, blocknum, packedUnit]
    // packedUnit: [ic/lp*(kx*ky)/blocknum, hp, lp] + [hp] + [hp]

    for (int hU = 0; hU < ocDiv; ++hU) {
        auto huPtr = dstbuffer + hU * blockNum * (stride1 + 2 * UNIT * infoBytes);
        int scaleCount = ALIMIN(ocUp4 - hU * UNIT, UNIT);
        for (int bl = 0; bl < blockNum; ++bl) {
            auto blockPtr = huPtr + bl * (stride1 + 2 * scaleCount * infoBytes);
            memcpy(blockPtr, src0 + bl * stride1 + hU * stride0, stride1);
            memcpy(blockPtr + stride1, src1 + (bl * ocUp4 + hU * UNIT) * infoBytes, scaleCount * infoBytes);
            memcpy(blockPtr + stride1 + scaleCount * infoBytes, src2 + (bl * ocUp4 + hU * UNIT) * infoBytes, scaleCount * infoBytes);
        }
    }
}

static void _computeReorderQuantInfo(std::shared_ptr<XPUConvolution::ResourceInt8> resource, std::shared_ptr<ConvolutionCommon::Int8Common> quantCommon, int outputCount, int kernelCount, int pack, AutoStorage<int8_t>& reorderedQuantInfo, float* ikernelSum, int HP, bool realInt4OrInt8) {
    // Only used for dynamic quant:
    // copy gemm bias
    // copy/compute real dequant bias/scale
    // dequant weight kernel sum
    int ocUp4 = ROUND_UP(outputCount, pack);
    int ocUpHp = ROUND_UP(outputCount, HP);

    int blockNum = resource->mBlockNum;
    int scaleSize = blockNum * ocUp4; // pack size.
    int blockSize = kernelCount / blockNum;
    int originOffset = 0;
    if (quantCommon->canUseInt4) {
        originOffset = -8;
    }
    // Save weight quant alpha and zero: wf=alpha*wi+zero
    auto alphaPtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
    auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(alphaPtr) + scaleSize * QUANT_INFO_BYTES);
    if (outputCount % pack != 0) {
        ::memset(alphaPtr, 0, scaleSize * QUANT_INFO_BYTES);
        ::memset(biasPtr, 0, scaleSize * QUANT_INFO_BYTES);
    }
    auto quanInfoPtr = quantCommon->alpha.get();
    auto weightKernelSum = (float*)(resource->mWeightKernelSum->deviceId());
    ::memset(weightKernelSum, 0, resource->mWeightKernelSum->size());

    bool blockQuantInput = (resource->mWeightKernelSum->length(0) / QUANT_INFO_BYTES == ocUp4) ? false : true;
    int ocDiv4 = UP_DIV(outputCount, pack);
    // dstkernelsum: [hU,blocknum,min(hP, pack)]
    if (quantCommon->asymmetric) {
        for (int i = 0; i < outputCount; ++i) {
            float accum = 0.f;
            auto ocOutside = i / HP;
            auto ocInside = i % HP;
            int remain = ALIMIN(HP, ocUp4 - ocOutside * HP);
            for (int j = 0; j < blockNum; ++j) {
                int index = i * blockNum + j;
                int srcSumIndex = ocOutside * blockNum * HP + j * HP + ocInside; // ikernelsum: [hU,blocknum,hP]
                alphaPtr[j * ocUp4 + i] = quanInfoPtr[2 * index + 1];
                biasPtr[j * ocUp4 + i] = quanInfoPtr[2 * index] + (float)originOffset * quanInfoPtr[2 * index + 1];
                if (realInt4OrInt8) {
                    accum += (ikernelSum[srcSumIndex] * quanInfoPtr[2 * index + 1] + blockSize * biasPtr[j * ocUp4 + i]);
                } else {
                    accum += ((ikernelSum[srcSumIndex]  - blockSize * 8)* quanInfoPtr[2 * index + 1] + blockSize * quanInfoPtr[2 * index]);
                }
                if (blockQuantInput) {
                    int dstSumIndex = ocOutside * blockNum * HP + j * remain + ocInside;
                    weightKernelSum[dstSumIndex] = accum;
                    accum = 0;
                }
            }
            if (!blockQuantInput) {
                weightKernelSum[i] = accum;
            }
        }
    } else {
        for (int i = 0; i < outputCount; ++i) {
            float accum = 0.f;
            auto ocOutside = i / HP;
            auto ocInside = i % HP;
            int remain = ALIMIN(HP, ocUp4 - ocOutside * HP);
            for (int j = 0; j < blockNum; ++j) {
                int index = i * blockNum + j;
                int srcSumIndex = ocOutside * blockNum * HP + j * HP + ocInside; // ikernelsum: [hU,blocknum,hP]
                alphaPtr[j * ocUp4 + i] = quanInfoPtr[index];
                biasPtr[j * ocUp4 + i] = (float)originOffset * quanInfoPtr[index];
                if (realInt4OrInt8) {
                    accum += (ikernelSum[srcSumIndex] * quanInfoPtr[index] + blockSize * biasPtr[j * ocUp4 + i]);
                } else {
                    accum += ((ikernelSum[srcSumIndex]  - blockSize * 8) * quanInfoPtr[index]);
                }
                if (blockQuantInput) {
                    int dstSumIndex = ocOutside * blockNum * HP + j * remain + ocInside;
                    weightKernelSum[dstSumIndex] = accum;
                    accum = 0;
                }
            }
            if (!blockQuantInput) {
                weightKernelSum[i] = accum;
            }
        }
    }
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant) : ConvInt8TiledExecutor(backend, op) {
    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int kernelCount = mCommon->kernelX() * mCommon->kernelY();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();
    
    int blockNum = 1;
    if (quanCommon) {
        int dequantCnt = quanCommon->alphaSize;
        if (quanCommon->asymmetric) {
            dequantCnt /= 2;
        }
        blockNum = dequantCnt / oc;
    }

    // backend info
    auto core = static_cast<XPUBackend*>(backend)->int8Functions();
    auto gcore = static_cast<XPUBackend*>(backend)->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int pack = gcore->pack;

    // compute info
    int ocUp4 = ROUND_UP(oc, pack);
    int lU = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    int scaleSize = ocUp4 * blockNum;
    std::vector<int> shape = {blockNum, UP_DIV(oc, UNIT), lU, UNIT, SRC_UNIT};
    mResourceInt8.reset(new XPUConvolution::ResourceInt8);
    mResourceInt8->mWeightAsymmetricQuant = quanCommon ? quanCommon->asymmetric : false;
    mResourceInt8->mActBits = 8;
    mResourceInt8->mBlockNum = blockNum;
    if (quanCommon && quanCommon->canUseInt4) {
        shape[4] = SRC_UNIT / 2;
        mResourceInt8->mActBits = 4;
        mResourceInt8->mWeightAsymmetricQuant = true; // offset: 8 from uint8_t
    }
    if (isDynamicQuant) {
        mResourceInt8->mDynamicQuant = true;
        // Relu/Relu6 post parameters
        auto postPtr = getPostParameters();
        mResourceInt8->mReluThreshold.resize(2);
        mResourceInt8->mReluThreshold[0] = postPtr[2];
        mResourceInt8->mReluThreshold[1] = postPtr[3];
        if (gcore->bytes == 2) {
            gcore->MNNFp32ToLowp(mResourceInt8->mReluThreshold.data(), reinterpret_cast<int16_t*>(mResourceInt8->mReluThreshold.data()), 2);
        }
    }
    // buffer allocate
    auto quantlen = 2 * blockNum * ROUND_UP(oc, pack) * QUANT_INFO_BYTES;
    auto weightlen = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({weightlen + quantlen}));
    mResourceInt8->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUp4})); // float
    auto dynamicOption = static_cast<XPUBackend*>(backend)->getRuntime()->hint().dynamicQuantOption; // input ic block quant.
    if (dynamicOption != 2) {
        mResourceInt8->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({QUANT_INFO_BYTES * ocUp4}));
    } else {
        mResourceInt8->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({blockNum * QUANT_INFO_BYTES * ocUp4}));
    }

    auto res = backend->onAcquireBuffer(mResourceInt8->mOriginBias.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightKernelSum.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

    if (!res) {
        MNN_ERROR("weight acquire buffer error\n");
        return;
    }
    bool useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;
    if (useCachedMmap) {
        return;
    }
    
    // read weight, weight's scale&bias, convolution bias
    ::memset((float*)(mResourceInt8->mOriginBias->deviceId()), 0, ocUp4 * sizeof(float));
    if (!isDynamicQuant) {
        mResourceInt8->mDynamicQuant = false;

        const int8_t* weightNotPacked;
        std::shared_ptr<float> scaleAndBias(new float[ocUp4 * 2], [](void* ptr) {
            delete [] (float*)ptr;
        });
        memset(scaleAndBias.get(), 0, ocUp4 * 2 * sizeof(float));
        int weightSize;
        ConvolutionCommon::getConvInt8Parameters(op, quanCommon, backend, weightNotPacked, weightSize, (float*)scaleAndBias.get(), (int32_t*)(mResourceInt8->mOriginBias->deviceId()), ocUp4);
        initializeConvInt8QuantInfo(mResourceInt8, convOp);

        auto L = ic * kernelCount;
        auto ptrWeight = (int8_t*)weightNotPacked;
        auto ptrWeightScale = scaleAndBias.get();
        auto ptrWeightBias = ptrWeightScale + ocUp4;
        for (int i = 0; i < oc; i++) {
            int temp = 0;
            for (int j = 0; j < L; j++) {
                temp += (int)ptrWeight[j + i * L];
            }
            ((float*)(mResourceInt8->mWeightKernelSum->deviceId()))[i] = (temp * ptrWeightScale[i] + L * ptrWeightBias[i]);
#ifdef MNN_USE_SSE
            if (mResourceInt8->mUseConvQuan) {
                ((int32_t*)(mResourceInt8->mOriginBias->deviceId()))[i] -= 128 * temp;
            }
#endif
        }
        mMutableResource.reset(new MutableResourceInt8(mResourceInt8, backend, scaleAndBias.get()));

        // reorder weight&scale&bias
        int bytes = 4;
        auto resourceInt8 = mResourceInt8;
        auto reorderFunction = [weightNotPacked, weightlen, oc, ic, kernelCount, UNIT, SRC_UNIT, shape, resourceInt8, bytes, L, ocUp4, quantlen, scaleAndBias]()
        {
            AutoStorage<uint8_t> weightReordered(weightlen);
            if (!weightReordered.get()) {
                MNN_ERROR("Memory not enough for quant model weight reorder\n");
                return -1;
            }
            int32_t info[6] = {1, oc, ic, kernelCount, UNIT, SRC_UNIT};
            ConvInt8TiledExecutor::reorderWeight((uint8_t*)weightReordered.get(), (uint8_t*)weightNotPacked, info, 0);
            int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], quantlen/ (2 * QUANT_INFO_BYTES * 1)};
            ConvInt8TiledExecutor::packWeightAndQuantInfo((int8_t*)(resourceInt8->mWeightInt8->deviceId()), (int8_t*)weightReordered.get(), (int8_t*)scaleAndBias.get(), params, QUANT_INFO_BYTES);
            return 0;
        };
        // static_cast<XPUBackend*>(backend)->enqueueTask(reorderFunction);
        reorderFunction();

        // gemmInt8 kernel
        mGemmKernel = core->Int8GemmKernel;
#ifdef MNN_USE_SSE
        int actBits = convOp->symmetricQuan()->nbits();
        if (actBits <= 7) {
            mGemmKernel = core->Int8GemmKernelFast;
        }
#else
        if(convOp->symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE){
            mGemmKernel = core->Int8GemmKernelFast;
        }
#endif

        return; // offline quant
    }

    // dynamic quant
    bool directReadInt4weight = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && ROUND_UP(ic, SRC_UNIT) == ic);
#ifdef MNN_KLEIDIAI_ENABLED
    if(quanCommon->canUseInt4) {
        bool bFP16 = gcore->bytes == 2 ? true : false;
        bool bAsym = quanCommon->asymmetric;
        size_t blkSize = mResourceInt8->mBlockNum == 1 ? 0 : ic / mResourceInt8->mBlockNum;
        KleidiAI::AccelType accelType = KleidiAI::getQIntAccelType(4, bAsym, blkSize);

        if (!KleidiAI::mKaiInitialized) {
            KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo(), bFP16, false);
        }

        KleidiAI& kai = KleidiAI::getInstance();
        if(!kai.isLoaded(accelType)) {
            kai.setLoaded(accelType);
            kai.printInfo(accelType);
        }

        if(kai.canAccelerate(accelType)) {
            AutoStorage<int8_t> reorderedQuantInfo;
            reorderedQuantInfo.reset(2 * scaleSize * QUANT_INFO_BYTES + oc * QUANT_INFO_BYTES);
            if (reorderedQuantInfo.get() == nullptr) {
                MNN_ERROR("Memory not enough\n");
                return;
            }

            //Prepare scale and zero data.
            {
                int outputCount = convOp->common()->outputCount();
                int originOffset = -8;
                auto quanInfoPtr = quanCommon->alpha.get();
                auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
                auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
                auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(zeroPtr) + scaleSize * QUANT_INFO_BYTES);
                if (quanCommon->asymmetric) {
                    for (int i = 0; i < blockNum; ++i) {
                        auto dstScale = scalePtr + i * ocUp4;
                        auto dstZero  = zeroPtr + i * ocUp4;
                        for (int j = 0; j < outputCount; ++j) {
                            int scaleIndex = j * blockNum + i;
                            dstScale[j] = quanInfoPtr[2 * scaleIndex + 1];
                            dstZero[j] = quanInfoPtr[2 * scaleIndex] + (float)originOffset * dstScale[j];
                        }
                    }
                } else {
                    for (int i = 0; i < blockNum; ++i) {
                        auto dstScale = scalePtr + i * ocUp4;
                        auto dstZero  = zeroPtr + i * ocUp4;
                        for (int j = 0; j < outputCount; ++j) {
                            int scaleIndex = j * blockNum + i;
                            dstScale[j] = quanInfoPtr[scaleIndex];
                            dstZero[j] = (float)originOffset * dstScale[j];
                        }
                    }
                }
                ::memcpy(biasPtr, convOp->bias()->data(), oc * QUANT_INFO_BYTES);
            }

            mAccelType = accelType;
            int n = oc;
            int k = ic;
            int packedWeightSize = kai.getRhsPackedSize(mAccelType, n, k, blkSize);

            //Alloc packed weight tensor.
            mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
            bool success = backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

            if (!success) {
                MNN_ERROR("Out of static memory!\n");
                return;
            }

            size_t paraNum = scaleSize;
            float *scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
            float *zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
            float *biasPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + 2 * paraNum;
            //Reload some parameters to fit ukernels' layout.
            auto quanInfoPtr = quanCommon->alpha.get();
            auto alphaSize = quanCommon->alpha.size();
            if(bAsym) {
                for(int i = 0; i < paraNum; i++) {
                    if(i*2 >= alphaSize){
                        zeroPtr[i] = 0;
                        scalePtr[i] = 0;
                    }
                    else{
                        zeroPtr[i] = quanInfoPtr[i * 2];
                        scalePtr[i] = quanInfoPtr[i * 2 + 1];
                    }
                }
            } else {
                if(blkSize != 0) {
                    memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
                }
            }

            //Run rhs pack.
            auto weightPackedData = (uint8_t*)(mResourceInt8->mWeightInt8->deviceId());
            kai.runRhsPack(mAccelType, 1, n, k, blkSize, 0/*unused*/,
                           (uint8_t*)quanCommon->weight.get(),
                           (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
                           weightPackedData);
            return;
        }
    }
#endif
    auto target = mResourceInt8;
    // Save bias
    if (convOp->bias()) {
        ::memcpy((float*)(mResourceInt8->mOriginBias->deviceId()), convOp->bias()->data(), oc * sizeof(float));
    }
    auto function = [shape, UNIT, SRC_UNIT, quanCommon, weightlen, scaleSize, directReadInt4weight, blockNum, ic, oc, quantlen, kernelCount, pack, convOp, gcore, target]() -> int {
        auto sh = shape;
        AutoStorage<int8_t> weightReordered(weightlen);
        AutoStorage<int8_t> reorderedQuantInfo(2 * scaleSize * QUANT_INFO_BYTES);
        AutoStorage<int8_t> kernelsum(blockNum * ROUND_UP(oc, UNIT) * QUANT_INFO_BYTES);
        if (weightReordered.get() == nullptr || reorderedQuantInfo.get() == nullptr || kernelsum.get() == nullptr) {
            MNN_ERROR("Memory not enough\n");
            return -1;
        }
        /* 1. reorder weight */
        if (quanCommon->canUseInt4 && directReadInt4weight) {
            auto srcPtr = (uint8_t*)quanCommon->weight.get();
            auto dstPtr = (uint8_t*)weightReordered.get();
            ::memset(dstPtr, 0, weightlen);
            gcore->MNNReorderWeightInt4(dstPtr, srcPtr, sh.data(), sh.size(), (float*)kernelsum.get());
        } else { // int4 weight but oc/ic not packed
            auto weightLength = quanCommon->weight.size();
            int blocksize = ic * kernelCount / blockNum;
            int originOffset = 0;
            int32_t info[6] = {blockNum, oc, ic, kernelCount, UNIT, SRC_UNIT};
            if (quanCommon->canUseInt4) {
                originOffset = -8;
                auto srcPtr = reinterpret_cast<uint8_t*>(quanCommon->weight.get());
                std::vector<uint8_t> tmpWeight(weightLength * 2);
                for (int j = 0; j < oc; ++j) {
                    for (int k = 0; k < blockNum; ++k) {
                        for (int i = 0; i < blocksize; ++i) {
                            int index = j * blockNum * blocksize + k * blocksize + i;
                            uint8_t w_ = srcPtr[index / 2];
                            int truew = index % 2 ? (w_ & 0x0f) : (w_ >> 4);
                            tmpWeight[index] = truew;
                        }
                    }
                }
                AutoStorage<uint8_t> packedInt8weight(weightlen * 2);
                if (packedInt8weight.get() == nullptr) {
                    MNN_ERROR("Weight reorder memory not enough!\n");
                    return -1;
                }
                reorderWeight(packedInt8weight.get(), (uint8_t*)tmpWeight.data(), info, 0, (float*)kernelsum.get(), gcore->MNNSumWeightInt8);
                // pack two int4 to int8
                int leng = weightlen * 2;
                auto srcint4Ptr = (uint8_t*)packedInt8weight.get();
                auto dstint4Ptr = (uint8_t*)weightReordered.get();
                int permuteUnit = UNIT * SRC_UNIT;
                int halfPermuteStride = static_cast<int32_t>(permuteUnit / 2);
                for (int i = 0; i < leng / permuteUnit; ++i) {
                    auto src0 = srcint4Ptr + i * permuteUnit;
                    auto dst0 = dstint4Ptr + i * halfPermuteStride;
                    for (int j = 0; j < halfPermuteStride; ++j) {
                        int s0 = src0[j];
                        int s1 = src0[j + halfPermuteStride];
                        int d = (s0) * 16 + (s1);
                        dst0[j] = d;
                    }
                }
            } else { // int8 weight
                reorderWeight((uint8_t*)weightReordered.get(), (uint8_t*)quanCommon->weight.get(), info, 0, (float*)kernelsum.get(), gcore->MNNSumWeightInt8);
            }
        }
        /* 2. compute and order dequant scale&bias */
        bool notConvertInt4ToInt8 = true;
        if (quanCommon->canUseInt4 && !directReadInt4weight) {
            notConvertInt4ToInt8 = false;
        }
        _computeReorderQuantInfo(target, quanCommon, oc, kernelCount * ic, pack, reorderedQuantInfo, (float*)kernelsum.get(), UNIT, notConvertInt4ToInt8);
        /* 3. put weight and quantInfo together */
        int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], quantlen / (2 * QUANT_INFO_BYTES * blockNum)};
        ConvInt8TiledExecutor::packWeightAndQuantInfo((int8_t*)(target->mWeightInt8->deviceId()), (int8_t*)weightReordered.get(), reorderedQuantInfo.get(), params, QUANT_INFO_BYTES);

        return 0;
    };
    // static_cast<XPUBackend*>(backend)->enqueueTask(std::move(function));
    function();
}

DenseConvInt8TiledExecutor::DenseConvInt8TiledExecutor(Backend* backend, const Op* op, const DenseConvInt8TiledExecutor& exe)
    : ConvInt8TiledExecutor(backend, op, exe.mResourceInt8), mGemmKernel(exe.mGemmKernel) {
}

DenseConvInt8TiledExecutor::~DenseConvInt8TiledExecutor() {
    // Do nothing
}

bool DenseConvInt8TiledExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    if (nullptr == dst) {
        return true;
    }
    auto exe = new DenseConvInt8TiledExecutor(bn, op, *this);
    if (!exe->valid()) {
        return false;
    }
#ifdef MNN_KLEIDIAI_ENABLED
    exe->mAccelType = this->mAccelType;
#endif
    *dst = exe;
    return true;
}

void DenseConvInt8TiledExecutor::getPackParameter(int* Unit, int* srcUnit, int* DestUnit, const XPUCoreInt8Functions* core) {
    core->MNNGetGemmUnit(Unit, srcUnit, DestUnit);
}


ErrorCode DenseConvInt8TiledExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Initialize.
    mUseBatchQuan = false;
    mIm2ColBasedInt8 = true;

    auto option = static_cast<XPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption;
    int batch = inputs[0]->batch();
    int inC   = inputs[0]->channel();
    auto output = outputs[0];
    int inputPlane  = batch * inputs[0]->width() * inputs[0]->height();
    auto planeSize = output->width() * output->height() * output->batch();
    auto core = static_cast<XPUBackend*>(backend())->int8Functions();
    auto gcore =static_cast<XPUBackend*>(backend())->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int kernelCount = mCommon->kernelY() * mCommon->kernelX();
    bool fastway = (kernelCount == 1) && (output->width() == inputs[0]->width()) && (output->height() == inputs[0]->height()) && (mCommon->strideX() * mCommon->strideY()) == 1;
    if (inputPlane > 1) {
        mUseBatchQuan = true;
    }
    if (!fastway) { // general conv
        mIm2ColBasedInt8 = false;
        if (planeSize > 1) {
            mUseBatchQuan = true;
        }
        if (option == 1) { // lowest level.
            mIm2ColBasedInt8 = true;
            mUseBatchQuan = false;
        }
    }
    
    float weightBytes = mResourceInt8->mActBits == 4 ? 0.5 : 1;
    mBlockNum = mResourceInt8->mBlockNum;
    
#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if(mResourceInt8->mDynamicQuant && mResourceInt8->mActBits == 4 && kai.canAccelerate(mAccelType)) {
        MNN_ASSERT(kai.isLoaded(mAccelType));
        const size_t m = inputs[0]->batch(); //lhs vector number.
        const size_t n = outputs[0]->channel(); //rhs vector number.
        const size_t k = inputs[0]->channel(); //vector size.
        const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;
        
        int packedSize = kai.getLhsQuantedPackedSize(mAccelType, m, k, blkSize);
        int elementSize = kai.isHalf() ? sizeof(__fp16) : sizeof(float);
        if(m > 1 && !kai.isLinear()) {
            int srcSize = m * k * elementSize;
            int dstSize = m * n * elementSize;
            int extraSize = srcSize > dstSize ? srcSize : dstSize;
            packedSize += extraSize;
        }
        
        //Split mTempIm2ColBuffer as two parts for linear/tile transfer:
        //Part0: Lhs_packed.
        //Part1: Lhs/Dst before transfer.
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
        bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        if (!success) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
        
        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }
#endif
    XPUConvolution::onResize(inputs, outputs);
    if (mResourceInt8->mDynamicQuant == false) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
        if (!mMutableResource->mResource->mUseConvQuan) {
            // In some previous quantized models, input's scale already fused with weight's scale and output's scale.
            // So there is no need to read input's scale additionally.
            mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({1, DST_XUNIT * QUANT_INFO_BYTES}));
            auto success = backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
            if (!success) {
                return OUT_OF_MEMORY;
            }
        }
        mBlockNum = 1;
        mIm2ColBasedInt8 = true;
        mUseBatchQuan = false;
    }
    XPUConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core);
    // input scale buffer
    const int threads = static_cast<XPUBackend*>(backend())->threadNumber();

    // Im2col info
    int im2colBytes = 1;
    const int L2Size = 2048;
    int tileLimitByC = UP_DIV(L2Size, mIm2ColParamter.kernelCountUnit * SRC_UNIT);
    
    if (mIm2ColBasedInt8 == false) {
        im2colBytes = gcore->bytes;
        tileLimitByC = 1;
    }
    int ic = inputs[0]->channel();
    int tileLimit = 0;
    int outC    = output->channel();
    int outC4 = UP_DIV(outC, gcore->pack);
    auto kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    mSplitByOc = true;
    
    // flop and io
    float flop = gcore->bytes * planeSize * (ROUND_UP(output->channel(), gcore->pack) * kernelCountUnit * SRC_UNIT / 1024.0 / 1024.0 / 1024.0);
    float ios  = (((XPUBackend*)backend())->getTensorSize(outputs[0], true) + ((XPUBackend*)backend())->getTensorSize(inputs[0], true) + ((XPUBackend*)backend())->getTensorSize(mResourceInt8->mWeightInt8.get()) * weightBytes) / (1024.0 * 1024.0 * 1024.0);
    
    if (threads < planeSize) { // Thread split by output nhw.
        tileLimit = ALIMIN(tileLimitByC, UP_DIV(planeSize, threads));
        mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
        if (mTileCount > threads) {
            mSplitByOc = false;
        }
        
    }
    if (mSplitByOc) {
        // tileLimit = ALIMIN(tileLimitByC, planeSize);
        // mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        // auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        // mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
        // auto ocPerThread = UP_DIV(outC4, threads);
        // auto threadNeed = UP_DIV(outC4, ocPerThread);
        // int totalWork = outC4;
        // int part = 1;
        // if (UNIT > gcore->pack) { // AVX512:UNIT=64,pack=16
        //     MNN_ASSERT(UNIT % gcore->pack == 0);
        //     int ocDivUnit = UP_DIV(outC4 * gcore->pack, UNIT);
        //     ocPerThread = UP_DIV(ocDivUnit, threads);
        //     threadNeed  = UP_DIV(ocDivUnit, ocPerThread);
        //     totalWork = ocDivUnit;
        //     part = UNIT / gcore->pack;
        // }
        // mThreadNums = ALIMIN(threads, threadNeed);
        
        // mDivides.resize(threads+1);
        // mDivides[0] = 0;
        // static_cast<XPUBackend *>(backend())->computeDivideSizes(totalWork, mDivides.data() + 1, flop / ios);
        // for (int i = 0; i < mDivides.size(); ++i) {
        //     mDivides[i] *= part;
        // }
    }
    
    if (!mSplitByOc) {
        mThreadNums = ALIMIN(threads, mTileCount);
        mDivides.resize(threads+1);
        mDivides[0] = 0;
        static_cast<XPUBackend *>(backend())->computeDivideSizes(mTileCount, mDivides.data() + 1, flop / ios);
    }
    int ocUp4 = ROUND_UP(outC, gcore->pack);
    int k = mThreadNums;
    int workPT = DST_XUNIT * mIm2ColCount;
    if (mSplitByOc) {
        k = 1; // Use one thread to finish im2col.
        workPT = mTileCount * DST_XUNIT * mIm2ColCount;
    }
    
    // auto bufferAlloc = static_cast<XPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = XPUConvolutionTiledExecutor::computeBlitInfoSize(workPT, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, k);
    mBlitInfoStride = blitInfoSize.second;
    // mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    mBlitInfo = malloc(blitInfoSize.first);
    const int unitColBufferSize  = kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    const int colBufferSize       = unitColBufferSize * mIm2ColCount;

    if (!mSplitByOc) {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({threads, colBufferSize * im2colBytes}));
        // mTempSrcSum = bufferAlloc->alloc(threads * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
        mTempSrcSum = malloc(threads * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
    } else {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mTileCount, colBufferSize * im2colBytes}));
        // mTempSrcSum = bufferAlloc->alloc(mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
        mTempSrcSum = malloc(mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
    }
    auto success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success || mBlitInfo == nullptr || mTempSrcSum == nullptr) {
        return OUT_OF_MEMORY;
    }
    if (false == mResourceInt8->mDynamicQuant) {
        // bufferAlloc->free(mBlitInfo);
        free(mBlitInfo);
        // bufferAlloc->free(mTempSrcSum);
        free(mTempSrcSum);
        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        if (mBatchQuantInfo.get()) {
            backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }

#ifdef MNN_LOW_MEMORY
    { // Dynamic Quant kernels
        mGemmKernel = core->Int8GemmKernel;
        if (mResourceInt8->mActBits == 4) {
            mGemmKernel = core->Int8GemmKernel_W4;
        }
        mQuantFunc = core->MNNFloat2Int8;
        if (gcore->bytes == 2 && gcore->pack == 8) {
            mGemmKernel = core->MNNGemmInt8AddBiasScale_Unit_FP16;
            if (mResourceInt8->mActBits == 4) {
                mGemmKernel = core->MNNGemmInt8AddBiasScale_w4_Unit_FP16;
            }
            mQuantFunc = core->DynamicQuanInput_ARM82;
            mQuantAndReorderFunc = core->DynamicQuanInputAndReorder_ARM82;
            
        }
        // A axisSum kernel
        mSumByAxisLFunc = gcore->MNNSumByAxisLForMatmul_A;
    }
    
    mInputBlockNum = (option == 2) ? mBlockNum : 1;
    bool symmetricQuant = (option != 2 && mUseBatchQuan) ? true : false;

    int size = 0;
    if (!mUseBatchQuan) { // single quant
        if (mSplitByOc) {
            size = 2 * mInputBlockNum * ALIMIN(DST_XUNIT, planeSize) * QUANT_INFO_BYTES;
        } else {
            size = 2 * mInputBlockNum * mIm2ColCount * DST_XUNIT * QUANT_INFO_BYTES;
        }
    }
    if (mUseBatchQuan) {
        if (mIm2ColBasedInt8) {
            size = 2 * mInputBlockNum * inputPlane * QUANT_INFO_BYTES;
        } else if (!mSplitByOc){ // only threads buffer needed by this case
            size = 2 * mInputBlockNum * mIm2ColCount * DST_XUNIT * QUANT_INFO_BYTES;
        } else {
            size = 2 * mInputBlockNum * planeSize * QUANT_INFO_BYTES;
        }
    }
    if (symmetricQuant) { // symmetric quant
        size /= 2;
    }
    if (!mIm2ColBasedInt8 && !mSplitByOc) {
        mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({threads, size}));
    } else {
        mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({1, size})); // keep dimensions=2!
    }
    success &= backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);

    // Dynamic quant.
    // set im2col tensor info
    if (mIm2ColBasedInt8) {
        mQuantInput.reset((Tensor::createDevice<int8_t>({batch, mIm2ColParamter.ih, mIm2ColParamter.iw, ROUND_UP(inC, gcore->pack)})));
    } else if (!mSplitByOc){
        mQuantInput.reset((Tensor::createDevice<int8_t>({threads, colBufferSize * 1})));
    } else {
        mQuantInput.reset((Tensor::createDevice<int8_t>({mTileCount, colBufferSize * 1})));
    }
    success &= backend()->onAcquireBuffer(mQuantInput.get(), Backend::DYNAMIC);
    
    // set compute buffer
    int tempSize = threads * 2 * mInputBlockNum * inputPlane;
    if (!mIm2ColBasedInt8) {
        if (!mSplitByOc) {
            tempSize = threads * 2 * mInputBlockNum * DST_XUNIT * mIm2ColCount;
        } else {
            tempSize = threads * 2 * mInputBlockNum * ROUND_UP(planeSize, DST_XUNIT);
        }
    }
    if (symmetricQuant) { // symmetric batch quant.
        tempSize /= 2;
    }
    mSizeInputBlockQuant = tempSize / threads;
    // mTempMaxMinValueBuffer = bufferAlloc->alloc(tempSize * gcore->bytes);
    mTempMaxMinValueBuffer = malloc(tempSize * gcore->bytes);
    // mQScaleZero = bufferAlloc->alloc(tempSize * QUANT_INFO_BYTES);
    mQScaleZero = malloc(tempSize * QUANT_INFO_BYTES);

    if (mQScaleZero == nullptr) {
        return OUT_OF_MEMORY;
    }
    mToFuseInputbias2Bias = (!mUseBatchQuan && option != 2) ? true : false;
    if (mToFuseInputbias2Bias) { // input data has only one bias&scale
        if (mIm2ColBasedInt8) {
            // mBiasBufferFusedInputzero = bufferAlloc->alloc(ocUp4 * QUANT_INFO_BYTES);
            mBiasBufferFusedInputzero = malloc(ocUp4 * QUANT_INFO_BYTES);
        } else {
            // mBiasBufferFusedInputzero = bufferAlloc->alloc(threads *ocUp4 * QUANT_INFO_BYTES);
            mBiasBufferFusedInputzero = malloc(threads *ocUp4 * QUANT_INFO_BYTES);
        }
        if (mBiasBufferFusedInputzero == nullptr) {
            return OUT_OF_MEMORY;
        }
    }
    mAccumBuffer.reset(Tensor::createDevice<int32_t>({threads, DST_XUNIT * ALIMAX(UNIT, gcore->pack)}));
    success &= backend()->onAcquireBuffer(mAccumBuffer.get(), Backend::DYNAMIC);

    if (mBlockNum > 1 && kernelCount > 1) {
        if (mSplitByOc) {
            // mReorderBuffer = bufferAlloc->alloc(UP_DIV(planeSize, DST_XUNIT) * unitColBufferSize);
            mReorderBuffer = malloc(UP_DIV(planeSize, DST_XUNIT) * unitColBufferSize);
        } else {
            // mReorderBuffer = bufferAlloc->alloc(threads * colBufferSize);
            mReorderBuffer = malloc(threads * colBufferSize);
        }
        if (mReorderBuffer == nullptr) {
            return OUT_OF_MEMORY;
        }
    }

    if (!success || mTempMaxMinValueBuffer == nullptr) {
        return OUT_OF_MEMORY;
    }
    // bufferAlloc->free(mBlitInfo);
    free(mBlitInfo);
    // bufferAlloc->free(mTempSrcSum);
    free(mTempSrcSum);
    // bufferAlloc->free(mTempMaxMinValueBuffer);
    free(mTempMaxMinValueBuffer);
    // bufferAlloc->free(mQScaleZero);
    free(mQScaleZero);
    if (mBlockNum >1 && kernelCount > 1) {
        // bufferAlloc->free(mReorderBuffer);
        free(mReorderBuffer);
    }
    if (mToFuseInputbias2Bias) {
        // bufferAlloc->free(mBiasBufferFusedInputzero);
        free(mBiasBufferFusedInputzero);
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mQuantInput.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mAccumBuffer.get(), Backend::DYNAMIC);
    
    return NO_ERROR;
#else
    return NO_ERROR;
#endif
}

ErrorCode DenseConvInt8TiledExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<XPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<XPUBackend*>(backend())->functions();
    auto dynamicOption = static_cast<XPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption;

#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if(mResourceInt8->mDynamicQuant && mResourceInt8->mActBits == 4 && kai.canAccelerate(mAccelType)) {
        MNN_ASSERT(kai.isLoaded(mAccelType));
        const size_t m = input->batch(); //lhs vector number.
        const size_t n = output->channel(); //rhs vector number.
        const size_t k = input->channel(); //vector size.
        const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;

        bool bHalf = kai.isHalf();
        size_t elementSize = bHalf ? sizeof(__fp16) : sizeof(float);
        size_t lhsPackedSize = kai.getLhsQuantedPackedSize(mAccelType, m, k, blkSize);

        auto lhs = (uint8_t*)(input->deviceId());
        auto lhsPacked = (int8_t*)(mTempIm2ColBuffer->deviceId());
        auto rhsPacked = (uint8_t*)(mResourceInt8->mWeightInt8->deviceId());
        auto dst = (uint8_t*)(output->deviceId());

        uint8_t *linearLhs, *linearDst;
        if(m > 1 && !kai.isLinear()) {
            linearLhs = (uint8_t *)lhsPacked + lhsPackedSize;
            linearDst = linearLhs;
        } else {
            linearLhs = lhs;
            linearDst = dst;
        }

        int threadNum = static_cast<XPUBackend*>(backend())->threadNumber();
        int threadNeed, vecPerThread;

        //Dynamic quant pack lhs.
        if(m == 1) {
            kai.runLhsQuantPack(mAccelType, 1, k, blkSize, 1, linearLhs, lhsPacked);
        } else {
            if(!kai.isLinear()) {
                if(bHalf) {
                    KleidiAIUtil::transferNC4HW4ToNCHW((__fp16 *)lhs, (__fp16 *)linearLhs, m, k);
                } else {
                    KleidiAIUtil::transferNC4HW4ToNCHW((float *)lhs, (float *)linearLhs, m, k);
                }
            }

            vecPerThread = kai.getVecNumPerThread(m, threadNum, kai.getMr(mAccelType, m));
            threadNeed = m % vecPerThread == 0 ? m / vecPerThread : (m / vecPerThread + 1);
            size_t srcStride = vecPerThread * k * elementSize;

            auto BatchDynamicQuant = [=, &kai](int tId) {
                auto threadSrc = linearLhs + tId * srcStride;
                auto threadDst = lhsPacked + kai.getLhsQuantedPackedOffset(mAccelType, m, tId * vecPerThread, k, blkSize);
                int vecNum = (tId == threadNeed - 1) ? (m - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
                kai.runLhsQuantPack(mAccelType, vecNum, k, blkSize, kai.getMr(mAccelType, m), threadSrc, threadDst);
            };

            MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
                BatchDynamicQuant((int)tId);
            }
            MNN_CONCURRENCY_END();
        }

        //Run matmul.
        if(kai.bSupportSme2() && mAccelType == KleidiAI::AccelType::QI4_SYM_CHNLQT) {
            //SME prefer running on single thread to obtain better performance/power consumption ratio.
            threadNum = 1;
        }

        vecPerThread = kai.getVecNumPerThread(n, threadNum, kai.getNStep(mAccelType));
        threadNeed = n % vecPerThread == 0 ? n / vecPerThread : (n / vecPerThread + 1);

        auto ThreadFunction = [=, &kai](int tId) {
            auto threadRhsPacked = rhsPacked + kai.getRhsPackedOffset(mAccelType, tId * vecPerThread, k, blkSize);
            auto threadDst = linearDst + kai.getDstOffset(0, tId * vecPerThread, n, elementSize);
            int vecNum = (tId == threadNeed - 1) ? (n - vecPerThread * tId) : vecPerThread; //Last threadN may less than vecPerThread.
            float scalarMax = bHalf ? FLT16_MAX : FLT_MAX;
            kai.runMatmul(mAccelType, m, vecNum, k, blkSize, lhsPacked, threadRhsPacked, threadDst, n * elementSize, elementSize, scalarMax, -scalarMax);
        };

        MNN_CONCURRENCY_BEGIN(tId, threadNeed) {
            ThreadFunction((int)tId);
        }
        MNN_CONCURRENCY_END();

        if(m > 1 && !kai.isLinear()) {
            if(bHalf) {
                KleidiAIUtil::transferNCHWToNC4HW4((__fp16 *)linearDst, (__fp16 *)dst, m, n);
            } else {
                KleidiAIUtil::transferNCHWToNC4HW4((float *)linearDst, (float *)dst, m, n);
            }
        }

        return NO_ERROR;
    }
#endif

    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    auto blitProc = core->MNNPackC4Int8ForMatMul_A;
    const int plane                  = output->batch() * mIm2ColParamter.oh * mIm2ColParamter.ow;
    const int batch                  = input->batch();
    const int PackUnit               = gcore->pack;
    const int dstZStep               = plane * PackUnit;
    const int ocDiv4                 = UP_DIV(output->channel(), PackUnit);
    const int ocUp4                  = ROUND_UP(output->channel(), PackUnit);
    const auto kernelCountUnit       = mIm2ColParamter.kernelCountUnit;
    const auto unitColBufferSize  = kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    const auto colBufferSize       = unitColBufferSize * mIm2ColCount;
    const auto dstBytes               = static_cast<XPUBackend*>(backend())->getBytes(backend(), output);
    const int blockL                 = kernelCountUnit / mBlockNum; // source depthQuad for each block.
    const int kxky                   = mIm2ColParamter.kernelX * mIm2ColParamter.kernelY;
    const int blocklu                = blockL / kxky;                     // UP_DIV(ic,src_unit) per block
    float weightBytes                = 1.f;
    int weightStepY                = weightBytes * (UNIT * SRC_UNIT);
    int inputPlane                   = batch * input->width() * input->height();

    auto im2colPtr           = (int8_t*)mTempIm2ColBuffer->deviceId();
    if (SRC_UNIT > PackUnit) {
        memset(im2colPtr, 0, mTempIm2ColBuffer->size());
    }
    const auto weightDataPtr = (int8_t*)(mResourceInt8->mWeightInt8->deviceId());
    // auto srcKernelSumPtr     = (int8_t*)mTempSrcSum.ptr();
    auto srcKernelSumPtr     = (int8_t*)mTempSrcSum;
    auto im2colSrc = (uint8_t*)(input->deviceId());
    auto outputDataPtr = (int8_t*)(output->deviceId());
    uint8_t* biasPtr = nullptr;
    int32_t inputZeroPoint = 0;
    int im2colBytes = mIm2ColBasedInt8 == true ? 1 : gcore->bytes;

    if (nullptr != mMutableResource.get()) {
        biasPtr       = (uint8_t*)(mMutableResource->mBiasFloat->deviceId());
        inputZeroPoint  = mMutableResource->mInputZeroPoint;
        if (mBatchQuantInfo.get()) {
            float scalein = TensorUtils::getQuantInfo(inputs[0])[0];
            float scaleou = TensorUtils::getQuantInfo(outputs[0])[0];
            auto scaleX = scalein / scaleou;
            for (int i = 0; i < DST_XUNIT; ++i) {
                ((float*)(mBatchQuantInfo->deviceId()))[i] = scaleX;
            }
        }
    }

#ifdef MNN_LOW_MEMORY
    auto BatchAsyDynamicQuant = [&](uint8_t* floatPtr, int32_t& inputZero, uint8_t* inputDequantScale, int LDiv4, int eCount, int innerSide, int32_t availableThreads, int8_t* dstInt8, uint8_t* inputDequantBias, int tId) {
        // if mIm2ColBasedInt8=false, input shape: [kernelsize,mBlockNum,blocklu,EP,LP]
        // if mIm2ColBasedInt8=true,  input shape: [ic/pack,EP,pack]
        auto scalePtr = (float*)inputDequantScale;
        auto zeroPtr = (float*)inputDequantBias;
        int scaleCount = mSizeInputBlockQuant;
        int kernelsize = 1;
        if (!mIm2ColBasedInt8) {
            kernelsize = kxky;
        }

        // auto minPtr = mTempMaxMinValueBuffer.ptr() + tId * scaleCount * gcore->bytes;
        auto minPtr = (uint8_t*)mTempMaxMinValueBuffer + tId * scaleCount * gcore->bytes;
        // auto maxPtr = mTempMaxMinValueBuffer.ptr() + tId * scaleCount * gcore->bytes + (scaleCount / 2) * gcore->bytes;
        auto maxPtr = (uint8_t*)mTempMaxMinValueBuffer + tId * scaleCount * gcore->bytes + (scaleCount / 2) * gcore->bytes;
        // auto qscale = (float*)(mQScaleZero.ptr() + tId * scaleCount * QUANT_INFO_BYTES);
        auto qscale = (float*)((uint8_t*)mQScaleZero + tId * scaleCount * QUANT_INFO_BYTES);
        // auto qbias  = (float*)(mQScaleZero.ptr() + tId * scaleCount * QUANT_INFO_BYTES + (scaleCount / 2) * QUANT_INFO_BYTES);
        auto qbias  = (float*)((uint8_t*)mQScaleZero + tId * scaleCount * QUANT_INFO_BYTES + (scaleCount / 2) * QUANT_INFO_BYTES);

        size_t info[9] = {(size_t)mInputBlockNum, (size_t)eCount, (size_t)innerSide, (size_t)DST_XUNIT, (size_t)SRC_UNIT, (size_t)kernelsize, (size_t)blocklu, 0, 0};
        if (mIm2ColBasedInt8) {
            info[6] = LDiv4 / mInputBlockNum;
        }
        if (mToFuseInputbias2Bias) {
            info[7] = 1;
        }
        if (mIm2ColParamter.padX > 0 || mIm2ColParamter.padY > 0) {
            info[8] = 1;
        }
        // scale&bias:float32
        gcore->MNNAsyQuantInfo(scalePtr, zeroPtr, qscale, qbias, (float*)minPtr, (float*)maxPtr, (float*)floatPtr, info);

        // quant: float->int8_t
        if (!mToFuseInputbias2Bias) {
            gcore->MNNAsyQuantFunc(dstInt8, (float*)floatPtr, qscale, qbias, info);
        } else {
            auto sizeDiv4 = UP_DIV(eCount * LDiv4 * innerSide, PackUnit);
            mQuantFunc((float*)floatPtr, dstInt8, sizeDiv4, qscale, -128, 127, qbias, 0);
        }

        if (mToFuseInputbias2Bias) { // Decode
            inputZero = qbias[0];
            // auto updatedBiasPtr = (float*)(mBiasBufferFusedInputzero.ptr() + tId * ocUp4 * QUANT_INFO_BYTES);
            auto updatedBiasPtr = (float*)((uint8_t*)mBiasBufferFusedInputzero + tId * ocUp4 * QUANT_INFO_BYTES);
            auto matmulBiasPtr = (float*)(mResourceInt8->mOriginBias->deviceId());
            auto weightKernelSum = (float*)(mResourceInt8->mWeightKernelSum->deviceId());
            auto zero_ = -inputZero * scalePtr[0];
            gcore->MNNDynamicUpdateConvBiasScale(updatedBiasPtr, matmulBiasPtr, weightKernelSum, &zero_, UP_DIV(ocUp4, 4));
            biasPtr = (uint8_t*)updatedBiasPtr;
            auto unitsize = mBatchQuantInfo->length(1) / (2 * QUANT_INFO_BYTES);
            auto inputScale = scalePtr[0];
            for (int i = 0; i < unitsize; ++i) {
                ((float*)inputDequantScale)[i] = inputScale;
            }
        }
    };

    auto BatchSymDynamicQuant = [&](uint8_t* floatPtr, int32_t& inputZero, uint8_t* inputDequantScale, int LU, int EP, int LP, int32_t availableThreads, int8_t* dstInt8, int tId) {
        // auto quantPtr = mQScaleZero.ptr() + tId * mSizeInputBlockQuant * QUANT_INFO_BYTES;
        auto quantPtr = (uint8_t*)mQScaleZero + tId * mSizeInputBlockQuant * QUANT_INFO_BYTES;
        // auto maxPtr = mTempMaxMinValueBuffer.ptr() + tId * mSizeInputBlockQuant * gcore->bytes;
        auto maxPtr = (uint8_t*)mTempMaxMinValueBuffer + tId * mSizeInputBlockQuant * gcore->bytes;

        // compute sum and absmax
        int divlu = UP_DIV(LU, availableThreads);
        MNN_CONCURRENCY_BEGIN (tIdx, ALIMIN(availableThreads, UP_DIV(LU, divlu))) {
            auto exeLu = ALIMIN(divlu, LU - tIdx * divlu);
            auto batchMax = reinterpret_cast<float*>(maxPtr + tIdx * EP * gcore->bytes);
            auto ptr_     = reinterpret_cast<float*>(floatPtr + tIdx * divlu * gcore->bytes * EP * LP);
            gcore->MNNAbsMax((float*)ptr_, batchMax, exeLu, EP, LP);
        } MNN_CONCURRENCY_END();
        

        // Compute quant scale
        gcore->MNNQuantScale((float*)maxPtr, (float*)quantPtr, (float*)inputDequantScale, availableThreads, EP);

        // quant
        auto scale_ptr = reinterpret_cast<float*>(quantPtr);
        gcore->MNNDynamicQuant((float*)floatPtr, dstInt8, scale_ptr, LU, EP, LP, nullptr);
        inputZero = 0;
    };

    if (mResourceInt8->mDynamicQuant) {
        biasPtr = (uint8_t*)(mResourceInt8->mOriginBias->deviceId());
    }
    if (mIm2ColBasedInt8 && mResourceInt8->mDynamicQuant) {
        int icDiv4 = UP_DIV(input->channel(), PackUnit);
        if (mUseBatchQuan) {
            int availthreads = (icDiv4 > mThreadNums && inputPlane > 255 ) ? mThreadNums : 1;
            if (dynamicOption != 2) {
                BatchSymDynamicQuant((uint8_t*)(input->deviceId()), inputZeroPoint, (uint8_t*)(mBatchQuantInfo->deviceId()), icDiv4, inputPlane, PackUnit, availthreads, (int8_t*)(mQuantInput->deviceId()), 0);
            } else {
                BatchAsyDynamicQuant((uint8_t*)(input->deviceId()), inputZeroPoint, (uint8_t*)(mBatchQuantInfo->deviceId()), icDiv4, inputPlane, PackUnit, availthreads, (int8_t*)(mQuantInput->deviceId()), (uint8_t*)(mBatchQuantInfo->deviceId()) + mBatchQuantInfo->stride(0) / 2, 0);
            }
        } else {
            BatchAsyDynamicQuant((uint8_t*)(input->deviceId()), inputZeroPoint, (uint8_t*)(mBatchQuantInfo->deviceId()), icDiv4, inputPlane, PackUnit, 1, (int8_t*)(mQuantInput->deviceId()), (uint8_t*)(mBatchQuantInfo->deviceId()) + mBatchQuantInfo->stride(0) / 2, 0);
        }
        im2colSrc = (uint8_t*)(mQuantInput->deviceId());
    }
#endif
    if (mResourceInt8->mActBits == 4) {
        weightBytes   = 0.5;
        weightStepY /= 2;
    }
    int blockunit = ocUp4 * 2 * QUANT_INFO_BYTES + blockL * weightStepY * UP_DIV(output->channel(), UNIT);
    auto inputchannel = input->channel();
    SumByAxisParams sumParams;
    sumParams.oneScale = (mUseBatchQuan || dynamicOption == 2) ? 0 : 1;
    sumParams.SRC_UNIT = SRC_UNIT;
    sumParams.blockNum = mBlockNum;
    sumParams.DST_XUNIT = DST_XUNIT;
    sumParams.unitColBufferSize = unitColBufferSize;
    sumParams.kernelCountUnitDouble = kernelCountUnit;
    sumParams.valid = inputchannel % SRC_UNIT;
    sumParams.kernelxy = kxky;
    sumParams.LU = UP_DIV(inputchannel, SRC_UNIT);
    sumParams.inputBlock = (mInputBlockNum > 1) ? 1 : 0;

    auto tileSplitFunction = [&](int tId, int eStartIndex, int eEndIndex, int estep) {
        auto ocDivThread = ocDiv4;
        float* reluPtr = mResourceInt8->mReluThreshold.data();
        float* accumbuff = nullptr;
        uint8_t* inputScale = nullptr;
        uint8_t* inputBias = nullptr;
        uint8_t* ptrInputScale = nullptr;
        uint8_t* ptrInputBias = nullptr;
        if (mBatchQuantInfo.get()) {
            if (mIm2ColBasedInt8) {
                inputScale = (uint8_t*)(mBatchQuantInfo->deviceId());
                ptrInputScale = inputScale;
            }
            
            if (dynamicOption == 2 && mUseBatchQuan && mIm2ColBasedInt8) {
                inputBias = inputScale + mBatchQuantInfo->stride(0) / 2;
                ptrInputBias = inputBias;
            }
        }
        if (mBlockNum > 1) {
            accumbuff = reinterpret_cast<float*>((int8_t*)(mAccumBuffer->deviceId()) + tId * mAccumBuffer->stride(0) * sizeof(int32_t));
        }
        float* ptrY = nullptr;
        if ((dstBytes != 1)) {
            ptrY = (float*)(mResourceInt8->mWeightKernelSum->deviceId());
        }
        QuanPostTreatParameters quanParam;
        quanParam.blockNum = mBlockNum;
        quanParam.weightKernelSum = ptrY;
        if (dstBytes != 1) {
            quanParam.useInt8 = 0;
            quanParam.fp32minmax = reluPtr;
        } else {
            quanParam.maxValue = mMutableResource->mClampMax;
            if (mResourceInt8->mRelu) {
                quanParam.minValue = mMutableResource->mOutputZeroPoint;
            } else {
                quanParam.minValue = mMutableResource->mClampMin;
            }
        }
        auto weightPtrTid = weightDataPtr;
        quanParam.biasFloat = reinterpret_cast<float*>(biasPtr);
        auto im2colDstThread        = im2colPtr + tId * mTempIm2ColBuffer->stride(0);
        // auto srcPtr     = (int8_t const **)(mBlitInfo.ptr() + tId * mBlitInfoStride.first);
        auto srcPtr     = (int8_t const **)((uint8_t* )mBlitInfo + tId * mBlitInfoStride.first);
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);
        auto xKernelSumPtrTid = reinterpret_cast<float*>(srcKernelSumPtr + tId * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);

        int32_t info[5];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(unitColBufferSize);
        info[3] = mIm2ColParamter.strideX;
        for (int tIndex = eStartIndex; tIndex < eEndIndex; tIndex += estep) {
            const int xIndexStart  = tIndex * DST_XUNIT * mIm2ColCount;
            auto outputInTilePtr = outputDataPtr + xIndexStart * PackUnit * dstBytes;
            int realDstCount = ALIMIN(plane - xIndexStart, DST_XUNIT * mIm2ColCount);
            ptrInputScale = (mUseBatchQuan && mIm2ColBasedInt8) ? (inputScale + xIndexStart * mInputBlockNum * QUANT_INFO_BYTES) : inputScale;
            ptrInputBias = (inputBias != nullptr) ? (inputBias + xIndexStart * mInputBlockNum * QUANT_INFO_BYTES) : inputBias;
            // im2col
            auto im2colDst = im2colDstThread;
            auto res = XPUConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, xIndexStart, realDstCount, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
            int number = res.first;
            bool needZero = res.second;
            if (needZero && mIm2ColBasedInt8) {
#ifdef MNN_USE_SSE
                ::memset(im2colDst, inputZeroPoint + 128, colBufferSize);
#else
                ::memset(im2colDst, inputZeroPoint, colBufferSize);
#endif
            }
            info[0] = number;
            info[4] = realDstCount;
            if (mIm2ColBasedInt8 && number > 0) {
                blitProc(im2colDst, srcPtr, info, el);
            }
#ifdef MNN_LOW_MEMORY
            if (!mIm2ColBasedInt8) {
                if (needZero) {
                    ::memset(im2colDst, 0, mTempIm2ColBuffer->stride(0));
                }
                if (number > 0) {
                    if (SRC_UNIT > PackUnit && !needZero) {
                        memset(im2colDst, 0, mTempIm2ColBuffer->stride(0));
                    }
                    info[2] = realDstCount;
                    gcore->MNNGeneralIm2Col((float*)im2colDst, (float const**)srcPtr, info, el, SRC_UNIT, PackUnit); // im2colDst: [lu, realDstCount, lp]
                }
                ptrInputScale = (uint8_t*)(mBatchQuantInfo->deviceId()) + tId * mBatchQuantInfo->stride(0);
                if (dynamicOption == 2) {
                    ptrInputBias = ptrInputScale + mBatchQuantInfo->stride(0) / 2;
                    BatchAsyDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputScale, kernelCountUnit, realDstCount, SRC_UNIT, 1, (int8_t*)(mQuantInput->deviceId()) + tId * mQuantInput->stride(0), ptrInputBias, tId);
                } else if (mUseBatchQuan) {
                    BatchSymDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputScale, kernelCountUnit, realDstCount, SRC_UNIT, 1, (int8_t*)(mQuantInput->deviceId()) + tId * mQuantInput->stride(0), tId);
                } else {
                    // auto maxMinPtr = mTempMaxMinValueBuffer.ptr() + tId * 2 * gcore->bytes;
                    auto maxMinPtr = (uint8_t*)mTempMaxMinValueBuffer + tId * 2 * gcore->bytes;
                    ptrInputBias = ptrInputScale + mBatchQuantInfo->stride(0) / 2;
                    BatchAsyDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputScale, kernelCountUnit, realDstCount, SRC_UNIT, 1, (int8_t*)(mQuantInput->deviceId()) + tId * mQuantInput->stride(0), ptrInputBias, tId);
                    // quanParam.biasFloat = (float*)(mBiasBufferFusedInputzero.ptr() + tId * ocUp4 * QUANT_INFO_BYTES);
                    quanParam.biasFloat = (float*)((uint8_t*)mBiasBufferFusedInputzero + tId * ocUp4 * QUANT_INFO_BYTES);
                }
                im2colDst = (int8_t*)(mQuantInput->deviceId()) + tId * mQuantInput->stride(0);
            }
            if (mBlockNum > 1 && kxky > 1) {
                auto eU = UP_DIV(realDstCount, DST_XUNIT); // eU <= mIm2ColCount
                // auto reorderBuffer = mReorderBuffer.ptr() + tId * colBufferSize;
                auto reorderBuffer = (uint8_t*)mReorderBuffer + tId * colBufferSize;
                for (int k = 0; k < eU; ++k) {
                    int inside  = blocklu * SRC_UNIT * ALIMIN(realDstCount - k * DST_XUNIT, DST_XUNIT);
                    auto dstbuffer = reorderBuffer + k * unitColBufferSize;
                    auto srcbuffer = im2colDst + k * unitColBufferSize;
                    for (int i = 0; i < mBlockNum; ++i) {
                        for (int j = 0; j < kxky; ++j) {
                            memcpy(dstbuffer + i * kxky * inside + j * inside, srcbuffer + i * inside + j * mBlockNum * inside, inside);
                        }
                    }
                }
                im2colDst = (int8_t*)reorderBuffer;
            }
#endif
            if (mResourceInt8->mWeightAsymmetricQuant) {
                MNN_ASSERT(mBatchQuantInfo.get() && (float*)(mBatchQuantInfo->deviceId()));
                gcore->MNNSumByAxisLForMatmul_A(xKernelSumPtrTid, im2colDst, (float*)ptrInputScale, realDstCount, sumParams);
            } else {
                memset(xKernelSumPtrTid, 0, mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
            }
            auto ptrX = xKernelSumPtrTid;
            do {
                int step = ALIMIN(DST_XUNIT, realDstCount);
                quanParam.inputScale = (float*)ptrInputScale;
                quanParam.inputBias = (float*)ptrInputBias;
                if (mBlockNum > 1) {
                    memset(accumbuff, 0, UNIT * 4 * DST_XUNIT);
                    quanParam.accumBuffer = accumbuff;
                }
                quanParam.srcKernelSum = ptrX;
                mGemmKernel(outputInTilePtr, im2colDst, weightPtrTid, blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
                
                ptrX += (step * mBlockNum);
                realDstCount-=step;
                outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                im2colDst += unitColBufferSize;
                ptrInputScale = mUseBatchQuan ? (ptrInputScale + step * mInputBlockNum * QUANT_INFO_BYTES) : ptrInputScale;
                ptrInputBias = (ptrInputBias != nullptr) ? (ptrInputBias + step * mInputBlockNum * QUANT_INFO_BYTES) : ptrInputBias;
            } while(realDstCount > 0);
        }
    };
    auto ocSplitFunction = [&](int threads) { // Thread split by OC
        auto im2colDst           = (int8_t*)(mTempIm2ColBuffer->deviceId());
        // auto srcPtr     = (int8_t const **)(mBlitInfo.ptr());
        auto srcPtr     = (int8_t const **)(mBlitInfo);
        auto el         = (int32_t *)(srcPtr + mBlitInfoStride.second);
        // auto xKernelSumPtr = reinterpret_cast<float*>(mTempSrcSum.ptr());
        auto xKernelSumPtr = reinterpret_cast<float*>(mTempSrcSum);

        auto eU = UP_DIV(plane, DST_XUNIT);
        int32_t info[5];
        info[1] = mIm2ColParamter.iw * mIm2ColParamter.ih * batch;
        info[2] = static_cast<int32_t>(unitColBufferSize);
        info[3] = mIm2ColParamter.strideX;
        
        float* reluPtr = mResourceInt8->mReluThreshold.data();
        if (mIm2ColBasedInt8) { // im2col
            auto res = XPUConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, 0, plane, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
            int number = res.first;
            bool needZero = res.second;
            if (needZero) {
#ifdef MNN_USE_SSE
                ::memset(im2colDst, inputZeroPoint + 128, mTempIm2ColBuffer->size());
#else
                ::memset(im2colDst, inputZeroPoint, mTempIm2ColBuffer->size());
#endif
            }
            info[0] = number;
            info[4] = plane;
            if (number > 0) {
                blitProc(im2colDst, srcPtr, info, el);
            }
        }
#ifdef MNN_LOW_MEMORY
        if (false == mIm2ColBasedInt8) {
            int realDstCount = plane;
            int start = 0;
            auto ptrInputscale = (uint8_t*)(mBatchQuantInfo->deviceId());
            auto ptrInputbias = ptrInputscale + mBatchQuantInfo->stride(0) / 2;
            auto int8Ptr = (int8_t*)(mQuantInput->deviceId());
            int sizePacked = 0;
            auto im2colDstTmp = im2colDst;
            while (realDstCount > 0) {
                int work = std::min(realDstCount, DST_XUNIT);
                sizePacked += (work * SRC_UNIT * kernelCountUnit);
                auto res = XPUConvolutionTiledExecutor::turnIm2ColToBlitInfo((const float**)srcPtr, el, start, work, mIm2ColParamter, (uint8_t*)im2colSrc, im2colBytes);
                int number = res.first;
                bool needZero = res.second;
                if (needZero) {
                    ::memset(im2colDstTmp, 0, unitColBufferSize * gcore->bytes);
                }
                info[0] = number;
                info[2] = work;
                if (number > 0) { // im2col
                    gcore->MNNGeneralIm2Col((float*)im2colDstTmp, (float const**)srcPtr, info, el, SRC_UNIT, PackUnit); // im2colDst: [lu, realDstCount, lp]
                }
                if (mUseBatchQuan || dynamicOption == 2) {
                    if (dynamicOption == 2) {
                        BatchAsyDynamicQuant((uint8_t*)im2colDstTmp, inputZeroPoint, ptrInputscale, kernelCountUnit, work, SRC_UNIT, 1, int8Ptr, ptrInputbias, 0);
                        ptrInputbias += (mInputBlockNum * work * sizeof(int32_t));
                    } else {
                        BatchSymDynamicQuant((uint8_t*)im2colDstTmp, inputZeroPoint, ptrInputscale, kernelCountUnit, work, SRC_UNIT, 1, int8Ptr, 0);
                    }
                    ptrInputscale += (mInputBlockNum * work * sizeof(int32_t));
                    int8Ptr += unitColBufferSize;
                }
                realDstCount -= work;
                start += work;
                im2colDstTmp += (unitColBufferSize * gcore->bytes);
            }
            if (!mUseBatchQuan && dynamicOption != 2) {
                BatchAsyDynamicQuant((uint8_t*)im2colDst, inputZeroPoint, ptrInputscale, kernelCountUnit, plane, SRC_UNIT, 1, (int8_t*)(mQuantInput->deviceId()), ptrInputscale + plane * mInputBlockNum* QUANT_INFO_BYTES, 0);
            }
            im2colDst = (int8_t*)(mQuantInput->deviceId());
        }
        if (mBlockNum > 1 && kxky > 1) {
            for (int k = 0; k < eU; ++k) {
                int inside  = blocklu * SRC_UNIT * ALIMIN(DST_XUNIT, plane - k * DST_XUNIT);
                // auto dstbuffer = mReorderBuffer.ptr() + k * unitColBufferSize;
                auto dstbuffer = (uint8_t*)mReorderBuffer + k * unitColBufferSize;
                auto srcbuffer = im2colDst + k * unitColBufferSize;
                for (int i = 0; i < mBlockNum; ++i) {
                    for (int j = 0; j < kxky; ++j) {
                        memcpy(dstbuffer + i * kxky * inside + j * inside, srcbuffer + i * inside + j * mBlockNum * inside, inside);
                    }
                }
            }
            // im2colDst = (int8_t*)mReorderBuffer.ptr();
            im2colDst = (int8_t*)mReorderBuffer;
        }
#endif
        if (mResourceInt8->mWeightAsymmetricQuant) {
            MNN_ASSERT(mBatchQuantInfo.get() && (float*)(mBatchQuantInfo->deviceId()));
            gcore->MNNSumByAxisLForMatmul_A(xKernelSumPtr, im2colDst, (float*)(mBatchQuantInfo->deviceId()), plane, sumParams);
        } else {
            memset(xKernelSumPtr, 0, mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
        }

        MNN_CONCURRENCY_BEGIN(tId, threads) {
            int ocIndex = PackUnit * mDivides[tId];
            auto ocDivThread = ALIMIN(mDivides[tId + 1] - mDivides[tId], ocDiv4 - mDivides[tId]);
            
            if (ocIndex < ocUp4) {
                auto im2colDstThread = im2colDst;
                float* ptrY = nullptr;
                if (dstBytes != 1) {
                    ptrY = (float*)(mResourceInt8->mWeightKernelSum->deviceId()) + (ocIndex / UNIT) * UNIT * mInputBlockNum;
                }
                QuanPostTreatParameters quanParam;
                quanParam.blockNum = mBlockNum;
                quanParam.weightKernelSum = ptrY;
                quanParam.biasFloat = reinterpret_cast<float*>(biasPtr + ocIndex * 4);
                if (dstBytes != 1) {
                    quanParam.useInt8 = 0;
                    quanParam.fp32minmax = reluPtr;
                } else {
                    quanParam.maxValue = mMutableResource->mClampMax;
                    if (mResourceInt8->mRelu) {
                        quanParam.minValue = mMutableResource->mOutputZeroPoint;
                    } else {
                        quanParam.minValue = mMutableResource->mClampMin;
                    }
                }
                uint8_t* inputScale = nullptr; // input scale for batch dynamic quant.
                uint8_t* inputBias = nullptr;
                float* accumbuff = nullptr;
                if (mBatchQuantInfo.get()) {
                    inputScale = (uint8_t*)(mBatchQuantInfo->deviceId());
                    if (dynamicOption == 2) {
                        inputBias = inputScale + mInputBlockNum * plane * QUANT_INFO_BYTES;
                    }
                }
                if (mBlockNum > 1) {
                    accumbuff = reinterpret_cast<float*>((int8_t*)(mAccumBuffer->deviceId()) + tId * mAccumBuffer->stride(0) * sizeof(int32_t));
                }

                auto outputInTilePtr = outputDataPtr + ocIndex * plane * dstBytes;
                const auto weightPtrTid = weightDataPtr + static_cast<int32_t>(ocIndex * mBlockNum * blockL * SRC_UNIT * weightBytes + ocIndex * 2 * mBlockNum * QUANT_INFO_BYTES);
                int realDstCount = plane;
                auto ptrX = xKernelSumPtr;
                do {
                    int step = ALIMIN(DST_XUNIT, realDstCount);
                    quanParam.inputScale = (float*)inputScale;
                    quanParam.inputBias = (float*)inputBias;
                    quanParam.srcKernelSum = ptrX;
                    if (mBlockNum > 1) {
                        memset(accumbuff, 0, UNIT * 4 * DST_XUNIT);
                        quanParam.accumBuffer = accumbuff;
                    }
                    mGemmKernel(outputInTilePtr, im2colDstThread, weightPtrTid, blockL, dstZStep * dstBytes, ocDivThread, &quanParam, step);
                    ptrX += (step * mBlockNum);
                    realDstCount-=step;
                    outputInTilePtr += DST_XUNIT * PackUnit * dstBytes;
                    im2colDstThread += unitColBufferSize;
                    inputScale = mUseBatchQuan ? (inputScale + mInputBlockNum * step * QUANT_INFO_BYTES) : inputScale;
                    inputBias = (inputBias != nullptr) ? (inputBias + mInputBlockNum * step * QUANT_INFO_BYTES) : inputBias;
                } while(realDstCount > 0);
            }
        }
        MNN_CONCURRENCY_END();
        
    };
    const int threads = static_cast<XPUBackend*>(backend())->threadNumber();
    if (!mSplitByOc) {
        MNN_CONCURRENCY_BEGIN(tId, threads) {
            if (mDivides[tId + 1] - mDivides[tId] > 0) {
                tileSplitFunction((int)tId, mDivides[tId], mDivides[tId + 1], 1);
            }
        }
        MNN_CONCURRENCY_END();
    } else {
        ocSplitFunction(threads);
    }
    return NO_ERROR;
}



// 计算输出特征图的高度和宽度
// inputH: 输入特征图高度, inputW: 输入特征图宽度, params: 卷积参数
static std::pair<int, int> calculateOutputSize(int inputH, int inputW,
                                        const Convolution2DCommonT &params) {
  // 核心公式：OH = ((IH + 2*padY - kernelY) / strideY) + 1
  int outputH = (inputH + 2 * params.padY - params.kernelY) / params.strideY + 1;
  // 核心公式：OW = ((IW + 2*padX - kernelX) / strideX) + 1
  int outputW = (inputW + 2 * params.padX - params.kernelX) / params.strideX + 1;
  return {outputH, outputW};
}

// 对输入特征图进行零填充（CAFFE模式默认零填充）
// input: 输入特征图 (N, C, H, W)，返回填充后的特征图 (N, C, H+2*padY, W+2*padX)
template <typename T>
std::vector<T> padInput(const std::vector<T> &input, int N, int C,
                            int H, int W, const Convolution2DCommonT &params) {
  int padLeft = params.pads[0];
  int padTop = params.pads[1];
  int padRight = params.pads[2];
  int padBottom = params.pads[3];

  // 填充后的高度和宽度
  int paddedH = H + padTop + padBottom;
  int paddedW = W + padLeft + padRight;

  // 初始化填充后的特征图，默认值为0（零填充）
  std::vector<T> paddedInput(N * C * paddedH * paddedW, 0.0f);

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
static std::vector<float> conv2d(const std::vector<float> &input,
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
      padInput<float>(input, N, inputC, inputH, inputW, params);
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

static std::vector<float> conv2d_int8(const std::vector<int8_t> &input,
                          const std::vector<int8_t> &weight,
                          const std::vector<float> &bias, 
                          const std::vector<float> &out_scales,
                          int N, int inputH,
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
  std::vector<int8_t> paddedInput =
      padInput<int8_t>(input, N, inputC, inputH, inputW, params);
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
                sum += (float)paddedInput[inputIdx] * (float)weight[weightIdx];
              }
            }
          }

          sum *= out_scales[oc];
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

DenseConvInt8TiledGeneralExecutor::DenseConvInt8TiledGeneralExecutor(Backend* backend, const Op* op, std::shared_ptr<ConvolutionCommon::Int8Common> quanCommon, bool isDynamicQuant) : ConvInt8TiledExecutor(backend, op) {
    
    conv_common_param_.padX        = mCommon->padX();
    conv_common_param_.padY        = mCommon->padY();
    conv_common_param_.padMode     = mCommon->padMode();
    conv_common_param_.kernelX     = mCommon->kernelX();
    conv_common_param_.kernelY     = mCommon->kernelY();
    conv_common_param_.strideX     = mCommon->strideX();
    conv_common_param_.strideY     = mCommon->strideY();
    conv_common_param_.dilateX     = mCommon->dilateX();
    conv_common_param_.dilateY     = mCommon->dilateY();
    conv_common_param_.group       = mCommon->group();
    conv_common_param_.outputCount = mCommon->outputCount();
    conv_common_param_.inputCount  = mCommon->inputCount();
    conv_common_param_.relu       = mCommon->relu();
    conv_common_param_.relu6      = mCommon->relu6();
    if (mCommon->pads()) {
        conv_common_param_.pads.assign(mCommon->pads()->begin(), mCommon->pads()->end());
    }
    if (mCommon->outPads()) {
        conv_common_param_.outPads.assign(mCommon->outPads()->begin(), mCommon->outPads()->end());
    }
    conv_common_param_.hasOutputShape = mCommon->hasOutputShape();


    // convolution info
    auto convOp = op->main_as_Convolution2D();
    int kernelCount = mCommon->kernelX() * mCommon->kernelY();
    int oc = convOp->common()->outputCount();
    int ic = convOp->common()->inputCount();

    conv_bias_.assign(convOp->bias()->data(), convOp->bias()->data() + oc);
    init_weight_.assign(quanCommon->weight.get(), quanCommon->weight.get() + quanCommon->weight.size());
    init_alpha_.assign(quanCommon->alpha.get(), quanCommon->alpha.get() + quanCommon->alpha.size());
    conv_weight_.resize(init_weight_.size());
    for (int i = 0; i < init_weight_.size(); ++i) {
        conv_weight_[i] = init_weight_[i] * init_alpha_[i/(ic*kernelCount)];
    }
    out_scales_.assign(init_alpha_.begin(), init_alpha_.end());
    for (int i = 0; i < out_scales_.size(); ++i) {
        out_scales_[i] =  out_scales_[i] * convOp->quanParameter()->scaleIn();
    }
    scale_in_ = convOp->quanParameter()->scaleIn();

    int blockNum = 1;
    if (quanCommon) {
        int dequantCnt = quanCommon->alphaSize;
        if (quanCommon->asymmetric) {
            dequantCnt /= 2;
        }
        blockNum = dequantCnt / oc;
    }

    // backend info
    auto core = static_cast<XPUBackend*>(backend)->int8Functions();
    auto gcore = static_cast<XPUBackend*>(backend)->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int pack = gcore->pack;

    // compute info
    int ocUp4 = ROUND_UP(oc, pack);
    int lU = UP_DIV(ic / blockNum, SRC_UNIT) * kernelCount;
    int scaleSize = ocUp4 * blockNum;
    std::vector<int> shape = {blockNum, UP_DIV(oc, UNIT), lU, UNIT, SRC_UNIT};
    mResourceInt8.reset(new XPUConvolution::ResourceInt8);
    mResourceInt8->mWeightAsymmetricQuant = quanCommon ? quanCommon->asymmetric : false;
    mResourceInt8->mActBits = 8;
    mResourceInt8->mBlockNum = blockNum;
    if (quanCommon && quanCommon->canUseInt4) {
        shape[4] = SRC_UNIT / 2;
        mResourceInt8->mActBits = 4;
        mResourceInt8->mWeightAsymmetricQuant = true; // offset: 8 from uint8_t
    }
    if (isDynamicQuant) {
        mResourceInt8->mDynamicQuant = true;
        // Relu/Relu6 post parameters
        auto postPtr = getPostParameters();
        mResourceInt8->mReluThreshold.resize(2);
        mResourceInt8->mReluThreshold[0] = postPtr[2];
        mResourceInt8->mReluThreshold[1] = postPtr[3];
        if (gcore->bytes == 2) {
            gcore->MNNFp32ToLowp(mResourceInt8->mReluThreshold.data(), reinterpret_cast<int16_t*>(mResourceInt8->mReluThreshold.data()), 2);
        }
    }
    // buffer allocate
    auto quantlen = 2 * blockNum * ROUND_UP(oc, pack) * QUANT_INFO_BYTES;
    auto weightlen = shape[0] * shape[1] * shape[2] * shape[3] * shape[4];
    mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({weightlen + quantlen}));
    mResourceInt8->mOriginBias.reset(Tensor::createDevice<int32_t>({ocUp4})); // float
    auto dynamicOption = static_cast<XPUBackend*>(backend)->getRuntime()->hint().dynamicQuantOption; // input ic block quant.
    if (dynamicOption != 2) {
        mResourceInt8->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({QUANT_INFO_BYTES * ocUp4}));
    } else {
        mResourceInt8->mWeightKernelSum.reset(Tensor::createDevice<uint8_t>({blockNum * QUANT_INFO_BYTES * ocUp4}));
    }

    auto res = backend->onAcquireBuffer(mResourceInt8->mOriginBias.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightKernelSum.get(), Backend::STATIC);
    res &= backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

    if (!res) {
        MNN_ERROR("weight acquire buffer error\n");
        return;
    }
    bool useCachedMmap = backend->getRuntime()->hint().useCachedMmap > 1;
    if (useCachedMmap) {
        return;
    }
    
    // read weight, weight's scale&bias, convolution bias
    ::memset((float*)(mResourceInt8->mOriginBias->deviceId()), 0, ocUp4 * sizeof(float));
    if (!isDynamicQuant) {
        mResourceInt8->mDynamicQuant = false;

        const int8_t* weightNotPacked;
        std::shared_ptr<float> scaleAndBias(new float[ocUp4 * 2], [](void* ptr) {
            delete [] (float*)ptr;
        });
        memset(scaleAndBias.get(), 0, ocUp4 * 2 * sizeof(float));
        int weightSize;
        ConvolutionCommon::getConvInt8Parameters(op, quanCommon, backend, weightNotPacked, weightSize, (float*)scaleAndBias.get(), (int32_t*)(mResourceInt8->mOriginBias->deviceId()), ocUp4);
        initializeConvInt8QuantInfo(mResourceInt8, convOp);

        auto L = ic * kernelCount;
        auto ptrWeight = (int8_t*)weightNotPacked;
        auto ptrWeightScale = scaleAndBias.get();
        auto ptrWeightBias = ptrWeightScale + ocUp4;
        for (int i = 0; i < oc; i++) {
            int temp = 0;
            for (int j = 0; j < L; j++) {
                temp += (int)ptrWeight[j + i * L];
            }
            ((float*)(mResourceInt8->mWeightKernelSum->deviceId()))[i] = (temp * ptrWeightScale[i] + L * ptrWeightBias[i]);
#ifdef MNN_USE_SSE
            if (mResourceInt8->mUseConvQuan) {
                ((int32_t*)(mResourceInt8->mOriginBias->deviceId()))[i] -= 128 * temp;
            }
#endif
        }
        mMutableResource.reset(new MutableResourceInt8(mResourceInt8, backend, scaleAndBias.get()));

        // reorder weight&scale&bias
        int bytes = 4;
        auto resourceInt8 = mResourceInt8;
        auto reorderFunction = [weightNotPacked, weightlen, oc, ic, kernelCount, UNIT, SRC_UNIT, shape, resourceInt8, bytes, L, ocUp4, quantlen, scaleAndBias]()
        {
            AutoStorage<uint8_t> weightReordered(weightlen);
            if (!weightReordered.get()) {
                MNN_ERROR("Memory not enough for quant model weight reorder\n");
                return -1;
            }
            int32_t info[6] = {1, oc, ic, kernelCount, UNIT, SRC_UNIT};
            ConvInt8TiledExecutor::reorderWeight((uint8_t*)weightReordered.get(), (uint8_t*)weightNotPacked, info, 0);
            int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], quantlen/ (2 * QUANT_INFO_BYTES * 1)};
            ConvInt8TiledExecutor::packWeightAndQuantInfo((int8_t*)(resourceInt8->mWeightInt8->deviceId()), (int8_t*)weightReordered.get(), (int8_t*)scaleAndBias.get(), params, QUANT_INFO_BYTES);
            return 0;
        };
        // static_cast<XPUBackend*>(backend)->enqueueTask(reorderFunction);
        reorderFunction();

        // gemmInt8 kernel
        mGemmKernel = core->Int8GemmKernel;
#ifdef MNN_USE_SSE
        int actBits = convOp->symmetricQuan()->nbits();
        if (actBits <= 7) {
            mGemmKernel = core->Int8GemmKernelFast;
        }
#else
        if(convOp->symmetricQuan()->method() == QuantizeAlgo_OVERFLOW_AWARE){
            mGemmKernel = core->Int8GemmKernelFast;
        }
#endif

        return; // offline quant
    }

    // dynamic quant
    bool directReadInt4weight = (kernelCount == 1 && ROUND_UP(oc, UNIT) == oc && ROUND_UP(ic, SRC_UNIT) == ic);
#ifdef MNN_KLEIDIAI_ENABLED
    if(quanCommon->canUseInt4) {
        bool bFP16 = gcore->bytes == 2 ? true : false;
        bool bAsym = quanCommon->asymmetric;
        size_t blkSize = mResourceInt8->mBlockNum == 1 ? 0 : ic / mResourceInt8->mBlockNum;
        KleidiAI::AccelType accelType = KleidiAI::getQIntAccelType(4, bAsym, blkSize);

        if (!KleidiAI::mKaiInitialized) {
            KleidiAI& kai = KleidiAI::getInstance(*MNNGetCPUInfo(), bFP16, false);
        }

        KleidiAI& kai = KleidiAI::getInstance();
        if(!kai.isLoaded(accelType)) {
            kai.setLoaded(accelType);
            kai.printInfo(accelType);
        }

        if(kai.canAccelerate(accelType)) {
            AutoStorage<int8_t> reorderedQuantInfo;
            reorderedQuantInfo.reset(2 * scaleSize * QUANT_INFO_BYTES + oc * QUANT_INFO_BYTES);
            if (reorderedQuantInfo.get() == nullptr) {
                MNN_ERROR("Memory not enough\n");
                return;
            }

            //Prepare scale and zero data.
            {
                int outputCount = convOp->common()->outputCount();
                int originOffset = -8;
                auto quanInfoPtr = quanCommon->alpha.get();
                auto scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
                auto zeroPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(scalePtr) + scaleSize * QUANT_INFO_BYTES);
                auto biasPtr = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(zeroPtr) + scaleSize * QUANT_INFO_BYTES);
                if (quanCommon->asymmetric) {
                    for (int i = 0; i < blockNum; ++i) {
                        auto dstScale = scalePtr + i * ocUp4;
                        auto dstZero  = zeroPtr + i * ocUp4;
                        for (int j = 0; j < outputCount; ++j) {
                            int scaleIndex = j * blockNum + i;
                            dstScale[j] = quanInfoPtr[2 * scaleIndex + 1];
                            dstZero[j] = quanInfoPtr[2 * scaleIndex] + (float)originOffset * dstScale[j];
                        }
                    }
                } else {
                    for (int i = 0; i < blockNum; ++i) {
                        auto dstScale = scalePtr + i * ocUp4;
                        auto dstZero  = zeroPtr + i * ocUp4;
                        for (int j = 0; j < outputCount; ++j) {
                            int scaleIndex = j * blockNum + i;
                            dstScale[j] = quanInfoPtr[scaleIndex];
                            dstZero[j] = (float)originOffset * dstScale[j];
                        }
                    }
                }
                ::memcpy(biasPtr, convOp->bias()->data(), oc * QUANT_INFO_BYTES);
            }

            mAccelType = accelType;
            int n = oc;
            int k = ic;
            int packedWeightSize = kai.getRhsPackedSize(mAccelType, n, k, blkSize);

            //Alloc packed weight tensor.
            mResourceInt8->mWeightInt8.reset(Tensor::createDevice<uint8_t>({packedWeightSize}));
            bool success = backend->onAcquireBuffer(mResourceInt8->mWeightInt8.get(), Backend::STATIC);

            if (!success) {
                MNN_ERROR("Out of static memory!\n");
                return;
            }

            size_t paraNum = scaleSize;
            float *scalePtr = reinterpret_cast<float*>(reorderedQuantInfo.get());
            float *zeroPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + paraNum;
            float *biasPtr = reinterpret_cast<float*>(reorderedQuantInfo.get()) + 2 * paraNum;
            //Reload some parameters to fit ukernels' layout.
            auto quanInfoPtr = quanCommon->alpha.get();
            auto alphaSize = quanCommon->alpha.size();
            if(bAsym) {
                for(int i = 0; i < paraNum; i++) {
                    if(i*2 >= alphaSize){
                        zeroPtr[i] = 0;
                        scalePtr[i] = 0;
                    }
                    else{
                        zeroPtr[i] = quanInfoPtr[i * 2];
                        scalePtr[i] = quanInfoPtr[i * 2 + 1];
                    }
                }
            } else {
                if(blkSize != 0) {
                    memcpy(scalePtr, (uint8_t*)quanInfoPtr, paraNum * sizeof(float));
                }
            }

            //Run rhs pack.
            auto weightPackedData = (uint8_t*)(mResourceInt8->mWeightInt8->deviceId());
            kai.runRhsPack(mAccelType, 1, n, k, blkSize, 0: unused,
                           (uint8_t*)quanCommon->weight.get(),
                           (const void*)scalePtr, (const void*)zeroPtr, (const void*)biasPtr,
                           weightPackedData);
            return;
        }
    }
#endif
    auto target = mResourceInt8;
    // Save bias
    if (convOp->bias()) {
        ::memcpy((float*)(mResourceInt8->mOriginBias->deviceId()), convOp->bias()->data(), oc * sizeof(float));
    }
    auto function = [shape, UNIT, SRC_UNIT, quanCommon, weightlen, scaleSize, directReadInt4weight, blockNum, ic, oc, quantlen, kernelCount, pack, convOp, gcore, target]() -> int {
        auto sh = shape;
        AutoStorage<int8_t> weightReordered(weightlen);
        AutoStorage<int8_t> reorderedQuantInfo(2 * scaleSize * QUANT_INFO_BYTES);
        AutoStorage<int8_t> kernelsum(blockNum * ROUND_UP(oc, UNIT) * QUANT_INFO_BYTES);
        if (weightReordered.get() == nullptr || reorderedQuantInfo.get() == nullptr || kernelsum.get() == nullptr) {
            MNN_ERROR("Memory not enough\n");
            return -1;
        }
        // 1. reorder weight
        if (quanCommon->canUseInt4 && directReadInt4weight) {
            auto srcPtr = (uint8_t*)quanCommon->weight.get();
            auto dstPtr = (uint8_t*)weightReordered.get();
            ::memset(dstPtr, 0, weightlen);
            gcore->MNNReorderWeightInt4(dstPtr, srcPtr, sh.data(), sh.size(), (float*)kernelsum.get());
        } else { // int4 weight but oc/ic not packed
            auto weightLength = quanCommon->weight.size();
            int blocksize = ic * kernelCount / blockNum;
            int originOffset = 0;
            int32_t info[6] = {blockNum, oc, ic, kernelCount, UNIT, SRC_UNIT};
            if (quanCommon->canUseInt4) {
                originOffset = -8;
                auto srcPtr = reinterpret_cast<uint8_t*>(quanCommon->weight.get());
                std::vector<uint8_t> tmpWeight(weightLength * 2);
                for (int j = 0; j < oc; ++j) {
                    for (int k = 0; k < blockNum; ++k) {
                        for (int i = 0; i < blocksize; ++i) {
                            int index = j * blockNum * blocksize + k * blocksize + i;
                            uint8_t w_ = srcPtr[index / 2];
                            int truew = index % 2 ? (w_ & 0x0f) : (w_ >> 4);
                            tmpWeight[index] = truew;
                        }
                    }
                }
                AutoStorage<uint8_t> packedInt8weight(weightlen * 2);
                if (packedInt8weight.get() == nullptr) {
                    MNN_ERROR("Weight reorder memory not enough!\n");
                    return -1;
                }
                reorderWeight(packedInt8weight.get(), (uint8_t*)tmpWeight.data(), info, 0, (float*)kernelsum.get(), gcore->MNNSumWeightInt8);
                // pack two int4 to int8
                int leng = weightlen * 2;
                auto srcint4Ptr = (uint8_t*)packedInt8weight.get();
                auto dstint4Ptr = (uint8_t*)weightReordered.get();
                int permuteUnit = UNIT * SRC_UNIT;
                int halfPermuteStride = static_cast<int32_t>(permuteUnit / 2);
                for (int i = 0; i < leng / permuteUnit; ++i) {
                    auto src0 = srcint4Ptr + i * permuteUnit;
                    auto dst0 = dstint4Ptr + i * halfPermuteStride;
                    for (int j = 0; j < halfPermuteStride; ++j) {
                        int s0 = src0[j];
                        int s1 = src0[j + halfPermuteStride];
                        int d = (s0) * 16 + (s1);
                        dst0[j] = d;
                    }
                }
            } else { // int8 weight
                reorderWeight((uint8_t*)weightReordered.get(), (uint8_t*)quanCommon->weight.get(), info, 0, (float*)kernelsum.get(), gcore->MNNSumWeightInt8);
            }
        }
        // 2. compute and order dequant scale&bias
        bool notConvertInt4ToInt8 = true;
        if (quanCommon->canUseInt4 && !directReadInt4weight) {
            notConvertInt4ToInt8 = false;
        }
        _computeReorderQuantInfo(target, quanCommon, oc, kernelCount * ic, pack, reorderedQuantInfo, (float*)kernelsum.get(), UNIT, notConvertInt4ToInt8);
        // 3. put weight and quantInfo together
        int32_t params[6] = {shape[0], shape[1], shape[2], shape[3], shape[4], quantlen / (2 * QUANT_INFO_BYTES * blockNum)};
        ConvInt8TiledExecutor::packWeightAndQuantInfo((int8_t*)(target->mWeightInt8->deviceId()), (int8_t*)weightReordered.get(), reorderedQuantInfo.get(), params, QUANT_INFO_BYTES);

        return 0;
    };
    // static_cast<XPUBackend*>(backend)->enqueueTask(std::move(function));
    function();
}

DenseConvInt8TiledGeneralExecutor::DenseConvInt8TiledGeneralExecutor(Backend* backend, const Op* op, const DenseConvInt8TiledGeneralExecutor& exe)
    : ConvInt8TiledExecutor(backend, op, exe.mResourceInt8), mGemmKernel(exe.mGemmKernel) {
}

DenseConvInt8TiledGeneralExecutor::~DenseConvInt8TiledGeneralExecutor() {
    // Do nothing
}

bool DenseConvInt8TiledGeneralExecutor::onClone(Backend* bn, const Op* op, Execution** dst) {
    MNN_PRINT("[warn][warn][warn][warn][warn][warn][warn][warn][warn]\n");
    if (nullptr == dst) {
        return true;
    }
    auto exe = new DenseConvInt8TiledGeneralExecutor(bn, op, *this);
    if (!exe->valid()) {
        return false;
    }
#ifdef MNN_KLEIDIAI_ENABLED
    exe->mAccelType = this->mAccelType;
#endif
    *dst = exe;
    return true;
}

void DenseConvInt8TiledGeneralExecutor::getPackParameter(int* Unit, int* srcUnit, int* DestUnit, const XPUCoreInt8Functions* core) {
    MNN_PRINT("[warn][warn][warn][warn][warn][warn][warn][warn][warn]\n");
    core->MNNGetGemmUnit(Unit, srcUnit, DestUnit);
}


ErrorCode DenseConvInt8TiledGeneralExecutor::onResize(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    // Initialize.
    mUseBatchQuan = false;
    mIm2ColBasedInt8 = true;

    auto option = static_cast<XPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption;
    int batch = inputs[0]->batch();
    int inC   = inputs[0]->channel();
    auto output = outputs[0];
    int inputPlane  = batch * inputs[0]->width() * inputs[0]->height();
    auto planeSize = output->width() * output->height() * output->batch();
    auto core = static_cast<XPUBackend*>(backend())->int8Functions();
    auto gcore =static_cast<XPUBackend*>(backend())->functions();
    int UNIT, SRC_UNIT, DST_XUNIT;
    core->MNNGetGemmUnit(&UNIT, &SRC_UNIT, &DST_XUNIT);
    int kernelCount = mCommon->kernelY() * mCommon->kernelX();
    bool fastway = (kernelCount == 1) && (output->width() == inputs[0]->width()) && (output->height() == inputs[0]->height()) && (mCommon->strideX() * mCommon->strideY()) == 1;
    if (inputPlane > 1) {
        mUseBatchQuan = true;
    }
    if (!fastway) { // general conv
        mIm2ColBasedInt8 = false;
        if (planeSize > 1) {
            mUseBatchQuan = true;
        }
        if (option == 1) { // lowest level.
            mIm2ColBasedInt8 = true;
            mUseBatchQuan = false;
        }
    }
    
    float weightBytes = mResourceInt8->mActBits == 4 ? 0.5 : 1;
    mBlockNum = mResourceInt8->mBlockNum;
    
#ifdef MNN_KLEIDIAI_ENABLED
    KleidiAI& kai = KleidiAI::getInstance();
    if(mResourceInt8->mDynamicQuant && mResourceInt8->mActBits == 4 && kai.canAccelerate(mAccelType)) {
        MNN_ASSERT(kai.isLoaded(mAccelType));
        const size_t m = inputs[0]->batch(); //lhs vector number.
        const size_t n = outputs[0]->channel(); //rhs vector number.
        const size_t k = inputs[0]->channel(); //vector size.
        const size_t blkSize = mBlockNum == 1 ? 0 : k / mBlockNum;
        
        int packedSize = kai.getLhsQuantedPackedSize(mAccelType, m, k, blkSize);
        int elementSize = kai.isHalf() ? sizeof(__fp16) : sizeof(float);
        if(m > 1 && !kai.isLinear()) {
            int srcSize = m * k * elementSize;
            int dstSize = m * n * elementSize;
            int extraSize = srcSize > dstSize ? srcSize : dstSize;
            packedSize += extraSize;
        }
        
        //Split mTempIm2ColBuffer as two parts for linear/tile transfer:
        //Part0: Lhs_packed.
        //Part1: Lhs/Dst before transfer.
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({packedSize}));
        bool success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        if (!success) {
            MNN_ERROR("Out of dynamic memory!\n");
            return OUT_OF_MEMORY;
        }
        
        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        return NO_ERROR;
    }
#endif
    XPUConvolution::onResize(inputs, outputs);
    if (mResourceInt8->mDynamicQuant == false) {
        mMutableResource->updateInputOutputScale(TensorUtils::getQuantInfo(inputs[0]), TensorUtils::getQuantInfo(outputs[0]));
        if (!mMutableResource->mResource->mUseConvQuan) {
            // In some previous quantized models, input's scale already fused with weight's scale and output's scale.
            // So there is no need to read input's scale additionally.
            mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({1, DST_XUNIT * QUANT_INFO_BYTES}));
            auto success = backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
            if (!success) {
                return OUT_OF_MEMORY;
            }
        }
        mBlockNum = 1;
        mIm2ColBasedInt8 = true;
        mUseBatchQuan = false;
    }
    XPUConvolutionTiledExecutor::setIm2ColParameter(mIm2ColParamter, mCommon, inputs[0], outputs[0], mPadX, mPadY, gcore, core);
    // input scale buffer
    const int threads = static_cast<XPUBackend*>(backend())->threadNumber();

    // Im2col info
    int im2colBytes = 1;
    const int L2Size = 2048;
    int tileLimitByC = UP_DIV(L2Size, mIm2ColParamter.kernelCountUnit * SRC_UNIT);
    
    if (mIm2ColBasedInt8 == false) {
        im2colBytes = gcore->bytes;
        tileLimitByC = 1;
    }
    int ic = inputs[0]->channel();
    int tileLimit = 0;
    int outC    = output->channel();
    int outC4 = UP_DIV(outC, gcore->pack);
    auto kernelCountUnit = mIm2ColParamter.kernelCountUnit;
    mSplitByOc = true;
    
    // flop and io
    float flop = gcore->bytes * planeSize * (ROUND_UP(output->channel(), gcore->pack) * kernelCountUnit * SRC_UNIT / 1024.0 / 1024.0 / 1024.0);
    float ios  = (((XPUBackend*)backend())->getTensorSize(outputs[0], true) + ((XPUBackend*)backend())->getTensorSize(inputs[0], true) + ((XPUBackend*)backend())->getTensorSize(mResourceInt8->mWeightInt8.get()) * weightBytes) / (1024.0 * 1024.0 * 1024.0);
    
    if (threads < planeSize) { // Thread split by output nhw.
        tileLimit = ALIMIN(tileLimitByC, UP_DIV(planeSize, threads));
        mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
        if (mTileCount > threads) {
            mSplitByOc = false;
        }
        
    }
    if (mSplitByOc) {
        // tileLimit = ALIMIN(tileLimitByC, planeSize);
        // mIm2ColCount = UP_DIV(tileLimit, DST_XUNIT);
        // auto DynamicDestUnit = DST_XUNIT * mIm2ColCount;
        // mTileCount        = UP_DIV(planeSize, DynamicDestUnit);
        // auto ocPerThread = UP_DIV(outC4, threads);
        // auto threadNeed = UP_DIV(outC4, ocPerThread);
        // int totalWork = outC4;
        // int part = 1;
        // if (UNIT > gcore->pack) { // AVX512:UNIT=64,pack=16
        //     MNN_ASSERT(UNIT % gcore->pack == 0);
        //     int ocDivUnit = UP_DIV(outC4 * gcore->pack, UNIT);
        //     ocPerThread = UP_DIV(ocDivUnit, threads);
        //     threadNeed  = UP_DIV(ocDivUnit, ocPerThread);
        //     totalWork = ocDivUnit;
        //     part = UNIT / gcore->pack;
        // }
        // mThreadNums = ALIMIN(threads, threadNeed);
        
        // mDivides.resize(threads+1);
        // mDivides[0] = 0;
        // static_cast<XPUBackend *>(backend())->computeDivideSizes(totalWork, mDivides.data() + 1, flop / ios);
        // for (int i = 0; i < mDivides.size(); ++i) {
        //     mDivides[i] *= part;
        // }
    }
    
    if (!mSplitByOc) {
        mThreadNums = ALIMIN(threads, mTileCount);
        mDivides.resize(threads+1);
        mDivides[0] = 0;
        static_cast<XPUBackend *>(backend())->computeDivideSizes(mTileCount, mDivides.data() + 1, flop / ios);
    }
    int ocUp4 = ROUND_UP(outC, gcore->pack);
    int k = mThreadNums;
    int workPT = DST_XUNIT * mIm2ColCount;
    if (mSplitByOc) {
        k = 1; // Use one thread to finish im2col.
        workPT = mTileCount * DST_XUNIT * mIm2ColCount;
    }
    
    // auto bufferAlloc = static_cast<XPUBackend*>(backend())->getBufferAllocator();
    auto blitInfoSize = XPUConvolutionTiledExecutor::computeBlitInfoSize(workPT, mIm2ColParamter.ow, mIm2ColParamter.kernelX * mIm2ColParamter.kernelY, k);
    mBlitInfoStride = blitInfoSize.second;
    // mBlitInfo = bufferAlloc->alloc(blitInfoSize.first);
    mBlitInfo = malloc(blitInfoSize.first);
    const int unitColBufferSize  = kernelCountUnit * DST_XUNIT * SRC_UNIT * sizeof(int8_t);
    const int colBufferSize       = unitColBufferSize * mIm2ColCount;

    if (!mSplitByOc) {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({threads, colBufferSize * im2colBytes}));
        // mTempSrcSum = bufferAlloc->alloc(threads * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
        mTempSrcSum = malloc(threads * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
    } else {
        mTempIm2ColBuffer.reset(Tensor::createDevice<int8_t>({mTileCount, colBufferSize * im2colBytes}));
        // mTempSrcSum = bufferAlloc->alloc(mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
        mTempSrcSum = malloc(mTileCount * mBlockNum * DST_XUNIT * mIm2ColCount * QUANT_INFO_BYTES);
    }
    auto success = backend()->onAcquireBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    if (!success || mBlitInfo == nullptr || mTempSrcSum == nullptr) {
        return OUT_OF_MEMORY;
    }
    if (false == mResourceInt8->mDynamicQuant) {
        // bufferAlloc->free(mBlitInfo);
        free(mBlitInfo);
        // bufferAlloc->free(mTempSrcSum);
        free(mTempSrcSum);
        backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
        if (mBatchQuantInfo.get()) {
            backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
        }
        return NO_ERROR;
    }

#ifdef MNN_LOW_MEMORY
    { // Dynamic Quant kernels
        mGemmKernel = core->Int8GemmKernel;
        if (mResourceInt8->mActBits == 4) {
            mGemmKernel = core->Int8GemmKernel_W4;
        }
        mQuantFunc = core->MNNFloat2Int8;
        if (gcore->bytes == 2 && gcore->pack == 8) {
            mGemmKernel = core->MNNGemmInt8AddBiasScale_Unit_FP16;
            if (mResourceInt8->mActBits == 4) {
                mGemmKernel = core->MNNGemmInt8AddBiasScale_w4_Unit_FP16;
            }
            mQuantFunc = core->DynamicQuanInput_ARM82;
            mQuantAndReorderFunc = core->DynamicQuanInputAndReorder_ARM82;
            
        }
        // A axisSum kernel
        mSumByAxisLFunc = gcore->MNNSumByAxisLForMatmul_A;
    }
    
    mInputBlockNum = (option == 2) ? mBlockNum : 1;
    bool symmetricQuant = (option != 2 && mUseBatchQuan) ? true : false;

    int size = 0;
    if (!mUseBatchQuan) { // single quant
        if (mSplitByOc) {
            size = 2 * mInputBlockNum * ALIMIN(DST_XUNIT, planeSize) * QUANT_INFO_BYTES;
        } else {
            size = 2 * mInputBlockNum * mIm2ColCount * DST_XUNIT * QUANT_INFO_BYTES;
        }
    }
    if (mUseBatchQuan) {
        if (mIm2ColBasedInt8) {
            size = 2 * mInputBlockNum * inputPlane * QUANT_INFO_BYTES;
        } else if (!mSplitByOc){ // only threads buffer needed by this case
            size = 2 * mInputBlockNum * mIm2ColCount * DST_XUNIT * QUANT_INFO_BYTES;
        } else {
            size = 2 * mInputBlockNum * planeSize * QUANT_INFO_BYTES;
        }
    }
    if (symmetricQuant) { // symmetric quant
        size /= 2;
    }
    if (!mIm2ColBasedInt8 && !mSplitByOc) {
        mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({threads, size}));
    } else {
        mBatchQuantInfo.reset(Tensor::createDevice<int8_t>({1, size})); // keep dimensions=2!
    }
    success &= backend()->onAcquireBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);

    // Dynamic quant.
    // set im2col tensor info
    if (mIm2ColBasedInt8) {
        mQuantInput.reset((Tensor::createDevice<int8_t>({batch, mIm2ColParamter.ih, mIm2ColParamter.iw, ROUND_UP(inC, gcore->pack)})));
    } else if (!mSplitByOc){
        mQuantInput.reset((Tensor::createDevice<int8_t>({threads, colBufferSize * 1})));
    } else {
        mQuantInput.reset((Tensor::createDevice<int8_t>({mTileCount, colBufferSize * 1})));
    }
    success &= backend()->onAcquireBuffer(mQuantInput.get(), Backend::DYNAMIC);
    
    // set compute buffer
    int tempSize = threads * 2 * mInputBlockNum * inputPlane;
    if (!mIm2ColBasedInt8) {
        if (!mSplitByOc) {
            tempSize = threads * 2 * mInputBlockNum * DST_XUNIT * mIm2ColCount;
        } else {
            tempSize = threads * 2 * mInputBlockNum * ROUND_UP(planeSize, DST_XUNIT);
        }
    }
    if (symmetricQuant) { // symmetric batch quant.
        tempSize /= 2;
    }
    mSizeInputBlockQuant = tempSize / threads;
    // mTempMaxMinValueBuffer = bufferAlloc->alloc(tempSize * gcore->bytes);
    mTempMaxMinValueBuffer = malloc(tempSize * gcore->bytes);
    // mQScaleZero = bufferAlloc->alloc(tempSize * QUANT_INFO_BYTES);
    mQScaleZero = malloc(tempSize * QUANT_INFO_BYTES);

    if (mQScaleZero == nullptr) {
        return OUT_OF_MEMORY;
    }
    mToFuseInputbias2Bias = (!mUseBatchQuan && option != 2) ? true : false;
    if (mToFuseInputbias2Bias) { // input data has only one bias&scale
        if (mIm2ColBasedInt8) {
            // mBiasBufferFusedInputzero = bufferAlloc->alloc(ocUp4 * QUANT_INFO_BYTES);
            mBiasBufferFusedInputzero = malloc(ocUp4 * QUANT_INFO_BYTES);
        } else {
            // mBiasBufferFusedInputzero = bufferAlloc->alloc(threads *ocUp4 * QUANT_INFO_BYTES);
            mBiasBufferFusedInputzero = malloc(threads *ocUp4 * QUANT_INFO_BYTES);
        }
        if (mBiasBufferFusedInputzero == nullptr) {
            return OUT_OF_MEMORY;
        }
    }
    mAccumBuffer.reset(Tensor::createDevice<int32_t>({threads, DST_XUNIT * ALIMAX(UNIT, gcore->pack)}));
    success &= backend()->onAcquireBuffer(mAccumBuffer.get(), Backend::DYNAMIC);

    if (mBlockNum > 1 && kernelCount > 1) {
        if (mSplitByOc) {
            // mReorderBuffer = bufferAlloc->alloc(UP_DIV(planeSize, DST_XUNIT) * unitColBufferSize);
            mReorderBuffer = malloc(UP_DIV(planeSize, DST_XUNIT) * unitColBufferSize);
        } else {
            // mReorderBuffer = bufferAlloc->alloc(threads * colBufferSize);
            mReorderBuffer = malloc(threads * colBufferSize);
        }
        if (mReorderBuffer == nullptr) {
            return OUT_OF_MEMORY;
        }
    }

    if (!success || mTempMaxMinValueBuffer == nullptr) {
        return OUT_OF_MEMORY;
    }
    // bufferAlloc->free(mBlitInfo);
    free(mBlitInfo);
    // bufferAlloc->free(mTempSrcSum);
    free(mTempSrcSum);
    // bufferAlloc->free(mTempMaxMinValueBuffer);
    free(mTempMaxMinValueBuffer);
    // bufferAlloc->free(mQScaleZero);
    free(mQScaleZero);
    if (mBlockNum >1 && kernelCount > 1) {
        // bufferAlloc->free(mReorderBuffer);
        free(mReorderBuffer);
    }
    if (mToFuseInputbias2Bias) {
        // bufferAlloc->free(mBiasBufferFusedInputzero);
        free(mBiasBufferFusedInputzero);
    }

    backend()->onReleaseBuffer(mTempIm2ColBuffer.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mBatchQuantInfo.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mQuantInput.get(), Backend::DYNAMIC);
    backend()->onReleaseBuffer(mAccumBuffer.get(), Backend::DYNAMIC);
    
    return NO_ERROR;
#else
    return NO_ERROR;
#endif
}

ErrorCode DenseConvInt8TiledGeneralExecutor::onExecute(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs) {
    const auto input = inputs[0];
    auto output      = outputs[0];
    auto core = static_cast<XPUBackend*>(backend())->int8Functions();
    auto gcore = static_cast<XPUBackend*>(backend())->functions();
    auto dynamicOption = static_cast<XPUBackend*>(backend())->getRuntime()->hint().dynamicQuantOption;

    Tensor input_nchw;
    TensorUtils::copyShape(inputs[0], &input_nchw, true, true);
    TensorUtils::getDescribe(&input_nchw)->dimensionFormat = MNN::MNN_DATA_FORMAT_NCHW;
    backend()->onAcquireBuffer(&input_nchw, Backend::STATIC);
    MNN::XPUTensorConverter::convert(inputs[0], &input_nchw);
    conv_input_.resize(input_nchw.elementSize());
    memcpy((void*)conv_input_.data(), (void*)input_nchw.deviceId(), input_nchw.size());
    std::vector<int8_t> conv_input_int8_(conv_input_.size());
    std::vector<float> conv_output;
    int8_t max_temp = 0;
    if (scale_in_ != 0) {
        for (int i = 0; i < conv_input_.size(); ++i) {
            conv_input_int8_[i] = (int8_t)(conv_input_[i] / scale_in_);
            if (abs(conv_input_int8_[i]) > max_temp) {
                max_temp = abs(conv_input_int8_[i]);
                MNN_PRINT("max_temp: %d\n", max_temp);
            }
        }
        // auto conv_output_temp = conv2d(conv_input_, conv_weight_, conv_bias_, inputs[0]->batch(), inputs[0]->height(), inputs[0]->width(), conv_common_param_);
        conv_output = conv2d_int8(conv_input_int8_, init_weight_, conv_bias_, out_scales_, inputs[0]->batch(), inputs[0]->height(), inputs[0]->width(), conv_common_param_);
    } else {
        conv_output = conv2d(conv_input_, conv_weight_, conv_bias_, inputs[0]->batch(), inputs[0]->height(), inputs[0]->width(), conv_common_param_);
    }
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
    return NO_ERROR;
}



} // namespace XPU
} // namespace MNN
