//
//  CommonOptFunction.cpp
//  MNN
//
//  Created by MNN on 2018/09/06.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "XPUCommonOptFunction.hpp"
// #include "ConvOpt.h"
// #include "WinogradOptFunction.hpp"
// #include "Int8FunctionsOpt.h"
// #include "ImageProcessFunction.hpp"
#include <string.h>
#include <algorithm>
#include <cmath>
#include <math.h>
#include "math/Vec.hpp"
#include <vector>
// #include "../CPURuntime.hpp"
#include "core/MemoryFormater.h"
// TODO: Find better way to optimize it
// #include "../CPUBinary.hpp"
// #include "../CPUUnary.hpp"
// #include "../CPUPool.hpp"
#define PACK 4
#define FLOAT float
using Vec = MNN::Math::Vec<float, 4>;
// #include "../GridSampler.hpp"
// #ifdef MNN_LOW_MEMORY
// #ifdef __aarch64__
// #include "backend/cpu/arm/arm64/low_memory/MNNDynamicQuantFunctions.hpp"
// #endif
// #endif

namespace MNN {
namespace XPU {

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

void MNNGetMatMulPackMode(int* eP, int *lP, int* hP) {
    *eP = 16;
    *lP = 1;
    *hP = 4;
}

void MNNPackForMatMul_B(float* dest, const float* source, size_t h, size_t l, bool transpose) {
    auto hP = h / 4;
    auto hR = hP * 4;
    if (hR != h) {
        ::memset(dest, 0, UP_DIV(h, 4)*4*l*sizeof(float));
    }
    if (!transpose) {
        for (int y=0; y<hP; ++y) {
            auto destY = dest + y * 4 * l;
            auto sourceY = source + y * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, 4 * sizeof(float));
            }
        }
        auto hRemain = h - hR;
        if (hRemain > 0) {
            auto destY = dest + hP * 4 * l;
            auto sourceY = source + hP * 4;
            for (int x=0; x<l; ++x) {
                ::memcpy(destY + 4 * x, sourceY + x * h, hRemain * sizeof(float));
            }
        }
        return;
    }
    int offset[] = {
        (int)l,
        (int)l
    };
    MNNPackC4(dest, source, l, h, offset);
}

static void _MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, int aStride) {
    auto h = parameter[2];
    auto l = parameter[1];
    auto cStride = parameter[3] / sizeof(float);
    auto hRemain = parameter[4];
    auto bExtraStride = parameter[5] / sizeof(float);
    auto bStride = bExtraStride + l * 4;
    auto hC4 = UP_DIV(h, 4);
    for (int y=0; y<hC4; ++y) {
        ::memset(C + y * cStride, 0, eSize * 4 * sizeof(float));
    }
    float alpha = 1.0f;
    float beta = 0.0f;
    float minValue = -std::numeric_limits<float>().max();
    float maxValue = std::numeric_limits<float>().max();
    if (nullptr != postParameters) {
        minValue = postParameters[2];
        maxValue = postParameters[3];
        alpha = postParameters[0];
        beta = postParameters[1];
    }

    for (int x=0; x<eSize; ++x) {
        auto dst = C + 4 * x;
        auto src = A + x;
        for (int y=0; y<hC4; ++y) {
            auto dstY = dst + y * cStride;
            auto weight = B + y * bStride;
            float summer[4] = {
                0.0f,
                0.0f,
                0.0f,
                0.0f,
            };
            if (nullptr != bias) {
                for (int v=0; v<4; ++v) {
                    summer[v] = bias[4 * y + v];
                }
            }
            for (int z=0; z<l; ++z) {
                auto aZ = src + z * aStride;
                auto wZ = weight + z * 4;
                summer[0] += wZ[0] * aZ[0];
                summer[1] += wZ[1] * aZ[0];
                summer[2] += wZ[2] * aZ[0];
                summer[3] += wZ[3] * aZ[0];
            }
            for (int v=0; v<4; ++v) {
                auto dstValue = std::min(summer[v], maxValue);
                dstValue = std::max(dstValue, minValue);
                dstY[v] = dstValue;
            }
        }
    }
}

void MNNPackedMatMul(float* C, const float* A, const float* B, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    return _MNNPackedMatMulRemain(C, A, B, 16, parameter, postParameters, bias, 16);
}

void MNNPackedMatMulRemain(float* C, const float* A, const float* B, size_t eSize, const size_t* parameter, const float* postParameters, const float* bias, const float* k, const float* b) {
    auto aStride = parameter[0] / sizeof(float);
    _MNNPackedMatMulRemain(C, A, B, eSize, parameter, postParameters, bias, aStride);
}

void MNNPackC4ForMatMul_A(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        auto dest = destOrigin + lOffset * eDest + eOffset;
        auto source = sourceGroup[n];

        for (int y=0; y<e; ++y) {
            auto yR = y % eDest;
            for (int x=0; x<l; ++x) {
                auto xR = x % 4;
                auto xC = x / 4;
                dest[(x) * eDest + yR] = source[xC * eReal * 4 + y * 4 * offset + xR];
            }
        }
    }
}

void MNNTranspose32Bit(int32_t* dstO, const int32_t* srcO, int32_t* dim) {
    int w = dim[0];
    int h = dim[1];
    int srcStride = dim[2];
    int dstStride = dim[3];
    for (int i=0; i<h; ++i) {
        auto si = srcO + i;
        auto di = dstO + i * dstStride;
        for (int j=0; j<w; ++j) {
            auto sj = si + j * srcStride;
            auto dj = di + j;
            *dj = *sj;
        }
    }
}

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC4Common<float>(dst, src, area, depth, areaOffset);
}

static XPUCoreFunctions* gCoreFunction = nullptr;

void MNNXPUCoreFunctionInit() {
    gCoreFunction = new XPUCoreFunctions;
    // // fp8
    // gCoreFunction->MNNFp32ToFp8 = MNNFp32ToFp8;
    // gCoreFunction->MNNFp16ToFp8 = MNNFp16ToFp8;
    // gCoreFunction->MNNFp8ToFp32 = MNNFp8ToFp32;
    // gCoreFunction->MNNFp8ToFp16 = MNNFp8ToFp16;

    // // MatMul
    gCoreFunction->MNNGetMatMulPackMode = MNNGetMatMulPackMode;
    gCoreFunction->MNNPackC4ForMatMul_A = MNNPackC4ForMatMul_A;
    gCoreFunction->MNNPackForMatMul_B = MNNPackForMatMul_B;
    gCoreFunction->MNNPackedMatMul = MNNPackedMatMul;
    gCoreFunction->MNNPackedMatMulRemain = MNNPackedMatMulRemain;
    // gCoreFunction->MNNCountMaxMinValue = MNNCountMaxMinValue;
    // gCoreFunction->MNNGetSparseMatMulPackMode = MNNGetSparseMatMulPackMode;
    // gCoreFunction->MNNAdjustOptimalSparseKernel = _MNNAdjustOptimalSparseKernel;

    // gCoreFunction->MNNComputeMatMulForE_1 = MNNComputeMatMulForE_1;
    // gCoreFunction->MNNComputeMatMulForH_1 = MNNComputeMatMulForH_1;

    // // Lowp
    // gCoreFunction->MNNFp32ToLowp = nullptr;
    // gCoreFunction->MNNLowpToFp32 = nullptr;
    gCoreFunction->bytes = 4;// sizeof(float)

    // Packed Function
    gCoreFunction->pack = 4;
    // FIXME: MNNPackTranspose and MNNUnpackTranspose is reverted
//     gCoreFunction->MNNPackCUnit = MNNPackC4;
//     gCoreFunction->MNNUnpackCUnit = MNNUnpackC4;
//     gCoreFunction->MNNUnpackCUnitTranspose = MNNPackTranspose;
//     gCoreFunction->MNNPackCUnitTranspose = MNNUnpackTranspose;
//     gCoreFunction->MNNPackCUnitInt8 = decltype(gCoreFunction->MNNPackCUnitInt8)(MNNPackC4Uint8);
//     gCoreFunction->MNNUnpackCUnitInt8 = decltype(gCoreFunction->MNNUnpackCUnitInt8)(MNNUnpackC4Uint8);
//     gCoreFunction->MNNPackCUnitTransposeInt8 = decltype(gCoreFunction->MNNPackCUnitTransposeInt8)(MNNUnpackTransposeUint8);
//     gCoreFunction->MNNUnpackCUnitTransposeInt8 = decltype(gCoreFunction->MNNUnpackCUnitTransposeInt8)(MNNPackTransposeUint8);
//     gCoreFunction->MNNPackCUnitInt16 = MNNPackC4Int16;
//     gCoreFunction->MNNUnpackCUnitInt16 = MNNUnpackC4Int16;
//     gCoreFunction->MNNPackCUnitTransposeInt16 = MNNUnpackTransposeInt16;
//     gCoreFunction->MNNUnpackCUnitTransposeInt16 = MNNPackTransposeInt16;

//     gCoreFunction->MNNAxByClampBroadcastUnit = MNNAxByClampBroadcastUnit;
//     gCoreFunction->MNNConvRunForLineDepthwise = MNNConvRunForLineDepthwise;
//     gCoreFunction->MNNMatrixAdd = MNNMatrixAdd;
//     gCoreFunction->MNNMatrixSub = MNNMatrixSub;
//     gCoreFunction->MNNStrassenMergeCFunction = MNNStrassenMergeCFunction;
//     gCoreFunction->penalty = 1.5f;
//     gCoreFunction->MNNScaleAndAddBias = MNNScaleAndAddBias;
//     gCoreFunction->MNNGridSampleComputeCord = MNNGridSampleComputeCord;
//     gCoreFunction->MNNGridSampleInterp = MNNGridSampleInterp;
// #ifndef MNN_REDUCE_SIZE
//     gCoreFunction->MNNGridSampleInterpGrad = MNNGridSampleInterpGrad;
// #endif
//     gCoreFunction->MNNGridSampleComputeCord3D = MNNGridSampleComputeCord3D;
//     gCoreFunction->MNNGridSampleInterp3D = MNNGridSampleInterp3D;
//     gCoreFunction->MNNRoiPoolingMax = MNNRoiPoolingMax;
//     gCoreFunction->MNNRoiAlignMax = MNNRoiAlignMax;
//     gCoreFunction->MNNRoiAlignAvg = MNNRoiAlignAvg;
//     gCoreFunction->MNNAddC4WithStride = MNNAddC4WithStride;
//     gCoreFunction->MNNCopyC4WithStride = MNNCopyC4WithStride;

//     gCoreFunction->chooseWinoSourceTransformPack = WinogradFunction::chooseWinoSourceTransformPack;
//     gCoreFunction->chooseWinoSourceUnrollTransform = WinogradFunction::chooseSourceUnrollTransform;
//     gCoreFunction->chooseWinoDestUnrollTransform = WinogradFunction::chooseWinoDestUnrollTransform;
//     gCoreFunction->MNNDeconvRunForLineDepthwise = MNNDeconvRunForLineDepthwise;
//     gCoreFunction->MNNDeconvRunForUnitDepthWise = MNNDeconvRunForUnitDepthWise;
// #ifdef MNN_USE_NEON
//     gCoreFunction->MNNDepthwiseConvFastKernel = MNNDepthwiseConvFastKernel;
// #endif
//     gCoreFunction->MNNSelectBinaryFunctionForFloat = CPUBinary::selectForFloat;
//     gCoreFunction->MNNSelectUnaryFunctionForFloat = CPUUnary::selectForFloat;
// #ifdef MNN_SUPPORT_QUANT_EXTEND
//     gCoreFunction->MNNSelectUnaryFunctionForInt8 = CPUUnary::selectForInt8;
// #endif
//     gCoreFunction->MNNReluWithSlopeChannel = MNNReluWithSlopeChannel;
//     gCoreFunction->MNNPoolingAvg = (decltype(gCoreFunction->MNNPoolingAvg))(poolingAvg<float, Vec4, 4>);
//     // Set min value as 1 << 24
//     gCoreFunction->MNNPoolingMax = (decltype(gCoreFunction->MNNPoolingMax))(poolingMax<float, Vec4, 4, -16777216>);

//     gCoreFunction->MNNPoolingMaxWithRedice = (decltype(gCoreFunction->MNNPoolingMaxWithRedice))(poolingMaxWithRedice<float, -16777216>);
//     // ImageProcess Functions
//     gCoreFunction->MNNRGBAToBGRA = MNNRGBAToBGRA;
//     gCoreFunction->MNNNV21ToRGBA = MNNNV21ToRGBA;
//     gCoreFunction->MNNNV21ToRGB = MNNNV21ToRGB;
//     gCoreFunction->MNNNV21ToBGRA = MNNNV21ToBGRA;
//     gCoreFunction->MNNNV21ToBGR = MNNNV21ToBGR;
//     gCoreFunction->MNNC1ToFloatC1 = MNNC1ToFloatC1;
//     gCoreFunction->MNNC3ToFloatC3 = MNNC3ToFloatC3;
//     gCoreFunction->MNNC3ToFloatRGBA = MNNC3ToFloatRGBA;
//     gCoreFunction->MNNSamplerC4Nearest = MNNSamplerC4Nearest;
//     gCoreFunction->MNNSamplerC4Bilinear = MNNSamplerC4Bilinear;

//     gCoreFunction->MNN4BitcopyWithStride = MNN4BitcopyWithStride;
//     gCoreFunction->MNN1BitcopyWithStride = MNN1BitcopyWithStride;
//     gCoreFunction->MNN2BitcopyWithStride = MNN2BitcopyWithStride;
//     gCoreFunction->MNN4BitcopyFast = MNN4BitcopyFast;
//     gCoreFunction->MNN2BitcopyFast = MNN2BitcopyFast;
//     gCoreFunction->MNN1BitcopyFast = MNN1BitCopyFast;

//     gCoreFunction->MNNAccumulateSequenceNumber = MNNAccumulateSequenceNumber;

//     const MNNCPUInfo& gCPUInfo = *MNNGetCPUInfo();
//     gCoreFunction->supportFp16arith = gCPUInfo.fp16arith;
//     gCoreFunction->supportSDot = gCPUInfo.dot;
//     gCoreFunction->supportI8mm = gCPUInfo.i8mm;
//     gCoreFunction->MNNSumByAxisLForMatmul_A = MNNSumByAxisLForMatmul_A;
//     gCoreFunction->MNNReorderWeightInt4 = MNNReorderWeightInt4;
//     gCoreFunction->MNNSumWeightInt8  = MNNSumWeightInt8;
// #ifdef __aarch64__
//    if (gCoreFunction->supportSDot) {
//        gCoreFunction->MNNReorderWeightInt4 = MNNReorderWeightInt4Arm82;
//        gCoreFunction->MNNSumWeightInt8 = MNNSumWeightInt8Arm82;
//    }
//    if (gCoreFunction->supportI8mm) {
//        gCoreFunction->MNNReorderWeightInt4 = MNNReorderWeightInt4Arm86;
//        gCoreFunction->MNNSumWeightInt8 = MNNSumWeightInt8Arm86;

//    }
// #endif
// #ifdef MNN_CPU_WEIGHT_DEQUANT_GEMM
//     // Weight Dequant Gemm Kernels
//     gCoreFunction->MNNPackedMatMul_int8 = MNNPackedMatMul_int8;
//     gCoreFunction->MNNPackedMatMulRemain_int8 = MNNPackedMatMulRemain_int8;
// #endif
// #ifdef MNN_LOW_MEMORY
//     gCoreFunction->MNNAbsMax = MNNAbsMaxFP32;                      // abs max value for [icDiv4,plane,4] -> abs max:[plane]
//     gCoreFunction->MNNDynamicQuant = MNNDynamicQuantFP32;          // symmetric 'batch' quant for [icDiv4,plane,4]
//     gCoreFunction->MNNAsyQuantFunc = MNNAsyQuantFunc;              // asymmetric 'batch' quant for [icDiv4,plane,4]
//     gCoreFunction->MNNAsyQuantInfo = MNNAsyQuantInfo_FP32;              // asymmetric quant/dequant scale&bias for [icDiv4,plane,4] -> scale&bias:[blockNum,plane]
//     gCoreFunction->MNNQuantScale = MNNQuantScaleFP32;              // symmetric quant/dequant scale&bias for [icDiv4,plane,4] -> scale&bias:[plane]
//     gCoreFunction->MNNGeneralIm2Col = generalIm2col;               // Im2Col based on float data -> output:[eU,kernelsize,lU,ep,lp]
//     gCoreFunction->MNNDynamicUpdateConvBiasScale = MNNDynamicUpdateConvBiasScale;
// #ifdef __aarch64__
//     if (gCoreFunction->supportSDot) {
//         gCoreFunction->MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Arm82;
//     }
//     if (gCoreFunction->supportI8mm) {
//         gCoreFunction->MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Arm86;
//     }
// #endif
// #endif
//     MNNCoreInt8FunctionInit();
//     MNNFunctionInit();
}

XPUCoreFunctions* MNNGetXPUCoreFunctions() {
    return gCoreFunction;
}

}; // namespace XPU
} // namespace MNN