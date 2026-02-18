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

#ifdef MNN_LOW_MEMORY
void MNNQuantScaleFP32(float* absmax, float* quant_scale, float* dequant_scale, size_t thread, size_t batch) {
    for (int i = 0; i < batch; ++i) {
        auto absmaxPtr = absmax + i;
        float absVal = 0.f;
        for (int t = 0; t < thread; ++t) {
            absVal = std::max(absVal, absmaxPtr[t * batch]);
        }
        if (absVal < 1e-7) {
            quant_scale[i] = 1.f;
            dequant_scale[i] = 1.f;
        } else {
            quant_scale[i] = 127.0f / absVal;
            dequant_scale[i] = absVal / 127.0f;
        }
    }
}
#endif

#ifdef MNN_LOW_MEMORY
static void MNNAbsMaxFP32(const float* source, float* absmax, size_t src_depth_quad, size_t realSize, int pack) {
// #ifdef __aarch64__
//     if (pack == 4) {
//         MNNAbsMaxFP32_Pack4(source, absmax, src_depth_quad, realSize, pack);
//         return;
//     }
//     if (pack == 8) {
//         MNNAbsMaxFP32_Pack8(source, absmax, src_depth_quad, realSize, pack);
//         return;
//     }
// #endif
    // source: (ic/4, N, 4)
    auto srcStep = pack * realSize;
    for (int i = 0; i < realSize; ++i) {
        float absmaxVal = 0.f; // absmaxVal>=0
        for (int c = 0; c < src_depth_quad; ++c) {
            auto src = source + c * srcStep + i * pack;
            for (int k = 0; k < pack; ++k) {
                absmaxVal = std::max(absmaxVal, std::abs(src[k]));
            }
        }
        absmax[i] = absmaxVal;
    }
}

void MNNDynamicQuantFP32(const float* src, int8_t* dst, const float* scale, size_t src_depth_quad, size_t realSize, int pack, const float* bias = nullptr) {
// #ifdef __aarch64__
//     if (pack == 4) {
//         MNNDynamicQuantFP32_Pack4(src, dst, scale, src_depth_quad, realSize, nullptr, pack);
//         return;
//     }
//     if (pack == 8) {
//         MNNDynamicQuantFP32_Pack8(src, dst, scale, src_depth_quad, realSize, nullptr, pack);
//         return;
//     }
// #endif
#ifdef MNN_USE_SSE
    uint8_t* dstPtr = reinterpret_cast<uint8_t*>(dst);
    int offset = 128;
#else
    int8_t* dstPtr = dst;
    int offset = 0;
#endif
    for (int i = 0; i < realSize; ++i) {
        auto scaleVal = scale[i];
        for (int c = 0; c < src_depth_quad; ++c) {
            auto srcZ = src + c * pack * realSize + i * pack;
            auto dstZ = dstPtr + c * pack * realSize + i * pack;
            for (int k = 0; k < pack; ++k) {
                int val = (int)roundf(srcZ[k] * scaleVal);
                dstZ[k] = val + offset;
            }
        }
    }
}

#endif // MNN_LOW_MEMORY

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

template<typename T>
void MNNUnpackC4Common(T* dst, const T* src, size_t area, size_t depth, int* areaOffset) {
    int depthC4     = depth / 4;
    int depthRemain = depthC4 * 4;
    int remain      = depth - depthRemain;
    int z, x, y;
    const T* srcChannel[4];
    const T* srcOffset = src;
    for(z = 0; z < depthC4; ++z) {
        for(y = 0; y < 4; ++y) {
            auto dstZ = dst + (z * 4 + y) * areaOffset[1];
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dstZ[x] = srcChannel[y][0];
                srcChannel[y] += 4;
            }
        }
        srcOffset += areaOffset[0] * 4;
    }
    if(remain > 0){
        auto dstZ = dst + depthC4 * areaOffset[1] * 4;
        for(y = 0; y < remain; ++y) {
            srcChannel[y] = srcOffset + y;
            for(x = 0; x < area; ++x) {
                dstZ[x] = srcChannel[y][0];
                srcChannel[y] += 4;
            }
            dstZ += areaOffset[1];
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
#define UNIT 4
using Vec4 = MNN::Math::Vec<float, 4>;

void MNNReluWithSlopeChannel(float* dst, const float* src, const float* slope, size_t sizeQuad, size_t depthQuad) {
    for (int j = 0; j < depthQuad; j++) {
        const float* slopeZ = slope + 4 * j;
        const float* srcZ   = src + 4 * j * sizeQuad;
        float* dstZ         = dst + 4 * j * sizeQuad;
        for (int i = 0; i < sizeQuad; i++) {
            for (int c = 0; c < 4; c++) {
                if (srcZ[4 * i + c] < 0) {
                    dstZ[4 * i + c] = srcZ[4 * i + c] * slopeZ[c];
                } else {
                    dstZ[4 * i + c] = srcZ[4 * i + c];
                }
            }
        }
    }
}

void MNNPackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNPackC4Common<float>(dst, src, area, depth, areaOffset);
}

void MNNUnpackC4(float* dst, const float* src, size_t area, size_t depth, int* areaOffset) {
    MNNUnpackC4Common<float>(dst, src, area, depth, areaOffset);
}

void MNNExpC8(float* dest, const float* source, float* offset, const float* parameters, size_t countC8) {
    auto count = countC8 * 8;
    auto param = parameters[0];
    float xLimit = 87;
    float summer = offset[3];
    for (int i = 0; i < count; ++i) {
        auto x         = source[i] * offset[0] + offset[2];
        x = ALIMAX(x, -xLimit);
        x = ALIMIN(x, xLimit);
        int div        = (x * parameters[1]);
        int div2       = (div + 127) << 23;
        auto xReamin   = x - div * param;
        float expBasic = *(float*)(&div2);
        auto t = xReamin * 0.25f;
        auto expRemain =
        ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + 1.0f) * t +
            1.0f;
        expRemain = expRemain * expRemain;
        expRemain = expRemain * expRemain;
        dest[i] = expBasic * expRemain + offset[1];
        summer+= dest[i];
    }
    offset[3] = summer;
}
void MNNScaleAndAddBias(float* dst, const float* src, const float* bias, const float* alpha, size_t planeNumber,
                        size_t biasNumber) {
    for (int z = 0; z < biasNumber; ++z) {
        float* dstZ         = dst + planeNumber * 4 * z;
        const float* srcZ   = src + planeNumber * 4 * z;
        auto biasZ = Vec4::load(bias + 4 * z);
        auto alphaZ = Vec4::load(alpha + 4 * z);
        for (int p = 0; p < planeNumber; ++p) {
            float* dstX       = dstZ + 4 * p;
            const float* srcX = srcZ + 4 * p;
            Vec4::save(dstX, (Vec4::load(srcX) * alphaZ) + biasZ);
        }
    }
}

void MNNExp(float* dst, const float* src, float* offset, size_t dataSize) {
    int countC8        = static_cast<int32_t>(dataSize) / 8;
    int remain = static_cast<int32_t>(dataSize) % 8;
    static const float parameters[] = {
        (float)logf(2.0f), 1.0f / (float)logf(2.0f), 0.25f, 1.0f, 0.5f, 1.0f / 6.0f, 1.0f / 24.0f, 1.0f / 120.0f};
    if (countC8 > 0) {
        // Align to eight so asm is easier to write
        MNNExpC8(dst, src, offset, parameters, countC8);
    }
    if (remain > 0) {
        auto param = parameters[0];
        float xLimit = 87;
        float summer = offset[3];
        auto source = src + countC8 * 8;
        auto dest = dst + countC8 * 8;
        for (int i = 0; i < remain; ++i) {
            auto x         = source[i] * offset[0] + offset[2];
            x = ALIMAX(x, -xLimit);
            x = ALIMIN(x, xLimit);
            int div        = (x * parameters[1]);
            int div2       = (div + 127) << 23;
            auto xReamin   = x - div * param;
            float expBasic = *(float*)(&div2);
            auto t = xReamin * 0.25f;
            auto expRemain =
            ((((parameters[7] * t + parameters[6]) * t + parameters[5]) * t + parameters[4]) * t + 1.0f) * t +
                1.0f;
            expRemain = expRemain * expRemain;
            expRemain = expRemain * expRemain;
            dest[i] = expBasic * expRemain + offset[1];
            summer+= dest[i];
        }
        offset[3] = summer;
    }
}
void MNNTanh(float* dst, const float* src, size_t dataSize) {
    /* Origin Code
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        dst[i] = tanhf_poly(src[i]);
    }
     */
    float offset[4] = {
        -2.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; i++) {
        // outputData[i] = 1 - 2 / (expf(2 * inputData[i]) + 1);
        auto expX2 = dst[i];
        dst[i] = (1.0f - expX2) / (1.0f + expX2);
    }
}

void MNNReluWithSlope(float* dst, const float* src, size_t sizeQuad, float slope) {
    float slopeValue[4];
    for (int i=0; i<4; ++i) {
        slopeValue[i] = slope;
    }
    MNNReluWithSlopeChannel(dst, src, slopeValue, sizeQuad, 1);
}

void MNNReluWithSlopeCommon(float* dst, const float* src, size_t size, float slope) {
    int sizeQuad = static_cast<int32_t>(size) / 4;
    int remain = static_cast<int32_t>(size) % 4;
    if (sizeQuad > 0) {
        MNNReluWithSlope(dst, src, sizeQuad, slope);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        MNNReluWithSlope(outmp, intmp, 1, slope);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
}

void MNNHardSwishCommon(float* dst, const float* src, size_t size) {
    int sizeQuad = static_cast<int32_t>(size / 4);
    int remain = static_cast<int32_t>(size) % 4;
#undef MNN_USE_SSE
#ifdef MNN_USE_SSE
    if (sizeQuad > 0) {
        MNNHardSwish(dst, src, sizeQuad);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        MNNHardSwish(outmp, intmp, 1);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
#else
#ifdef MNN_USE_NEON
    float32x4_t zero = vdupq_n_f32(0.f);
    float32x4_t three = vdupq_n_f32(3.f);
    float32x4_t six = vdupq_n_f32(6.f);
    float32x4_t divsix = vdupq_n_f32(1.0f/6.f);
    for (int i = 0; i < sizeQuad; i++) {
        auto x = vld1q_f32(src + 4 * i);
        auto y = vmulq_f32(vmulq_f32(x, vminq_f32(vmaxq_f32(vaddq_f32(x, three), zero), six)), divsix);
        vst1q_f32(dst + 4 * i, y);
    }
    if (remain > 0) {
        float intmp[4] = {0}, outmp[4] = {0};
        ::memcpy(intmp, src + sizeQuad * 4, remain * sizeof(float));
        auto x = vld1q_f32(intmp);
        auto y = vmulq_f32(vmulq_f32(x, vminq_f32(vmaxq_f32(vaddq_f32(x, three), zero), six)), divsix);
        vst1q_f32(outmp, y);
        ::memcpy(dst + sizeQuad * 4, outmp, remain * sizeof(float));
    }
#else
    for (int j = 0; j < size; j++) {
        if (src[j] <= -3) {
            dst[j] = 0;
        } else if (src[j] >= 3){
            dst[j] = src[j];
        } else {
            dst[j] = src[j] * (src[j] + 3) / 6.f;
        }
    }
#endif
#endif
}

void MNNGeluStandardCommon(float* dst, const float* src, size_t size) {
    for (int i = 0; i < size; i++) {
        dst[i] = (erf(src[i] * 0.7071067932881648) + 1) * src[i] * 0.5;
    }
}

void MNNGeluCommon(float* dst, const float* src, size_t size) {
    int sizeQuad = static_cast<int32_t>(size / 8);
    int remain = static_cast<int32_t>(size) % 8;
#if defined(MNN_USE_SSE) || defined(MNN_USE_NEON)
    float parameters[8] = {0.044715f, 0.79788458f, 378.f, 17325.f, 135135.f, 28.f, 3150.f, 62370.f};
    if (sizeQuad > 0) {
        MNNGelu(dst, src, sizeQuad, parameters);
    }
    if (remain > 0) {
        float intmp[8] = {0};
        float outmp[8] = {0};
        ::memcpy(intmp, src + 8 * sizeQuad, remain * sizeof(float));
        MNNGelu(outmp, intmp, 1, parameters);
        ::memcpy(dst + 8 * sizeQuad, outmp, remain * sizeof(float));
    }
#else
    auto tanhf_poly = [](float value) -> float {
        if (value > 5.0f) {
            return 1.0f;
        } else if (value <= -5.0f) {
            return -1.0f;
        } else {
            float x2 = value * value;
            float a  = value * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
            float b  = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
            return a / b;
        }
    };
    for (int i = 0; i < size; i++) {
        float temp = 0.044715f * src[i] * src[i] * src[i];
        temp = 0.79788458f * (temp + src[i]);
        dst[i] = (1.0f + tanhf_poly(temp)) * src[i] * 0.5f;
    }
#endif
}

void MNNScaleAndAddBiasScalar(float* dst, const float* src, float bias, float alpha, size_t number) {
    int numberC4 = (int)number / 4;
    int start = 0;
    if (numberC4 > 0) {
        float biasC4[4] = {
            bias,
            bias,
            bias,
            bias
        };
        float alphaC4[4] = {
            alpha,
            alpha,
            alpha,
            alpha
        };
        MNNScaleAndAddBias(dst, src, biasC4, alphaC4, numberC4, 1);
        start = numberC4 * 4;
    }
    for (int i=start; i<number; ++i) {
        dst[i] = src[i] * alpha + bias;
    }
}
void MNNSin(float* dst, const float* src, size_t dataSize) {
    for (int i = 0; i < dataSize; i++) {
        dst[i] = sinf(src[i]);
    }
}
void MNNSigmoid(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
}

void MNNSiLu(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = src[i] / (1.0f + dst[i]);
    }
}

/**
 Modified from https://github.com/alibaba/MNN/pull/1359
 Thanks for https://github.com/hroken
 */
void MNNSigmoidLowp(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
#ifdef MNN_USE_NEON
    int dataC4 = static_cast<int32_t>(dataSize) / 4;
    int remain = static_cast<int32_t>(dataSize) % 4;
    float32x4_t value = vdupq_n_f32(1.0f);

    if(dataC4 > 0) {
        float32x4_t out = vld1q_f32(dst);
        // neon optimization for sigmid cpu
        for (int i = 1; i < dataC4; ++i) {
            out = vrecpeq_f32(vaddq_f32(value,out));
            vst1q_f32(dst ,out);
            dst += 4;
            out = vld1q_f32(dst);
        }
        out = vrecpeq_f32(vaddq_f32(value,out));
        vst1q_f32(dst, out);
        dst += 4;
    }
    if (remain > 0) {
        float intmp[4] = {0};
        ::memcpy(intmp, dst, remain * sizeof(float));
        float32x4_t out = vld1q_f32(intmp);
        out = vrecpeq_f32(vaddq_f32(value,out));
        vst1q_f32(intmp, out);
        ::memcpy(dst, intmp, remain * sizeof(float));
    }
#else
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = 1.0f / (1.0f + dst[i]);
    }
#endif
}

void MNNSiLuLowp(float* dst, const float* src, size_t dataSize) {
    float offset[4] = {
       -1.0f,
        0.0f,
        0.0f,
        0.0f
    };
    MNNExp(dst, src, offset, dataSize);
#ifdef __aarch64__
    int dataC4 = static_cast<int32_t>(dataSize) / 4;
    int remain = static_cast<int32_t>(dataSize) % 4;
    float32x4_t one = vdupq_n_f32(1.0f);

    if(dataC4 > 0) {
        float32x4_t out = vld1q_f32(dst);
        float32x4_t in = vld1q_f32(src);
        // neon optimization for sigmid cpu
        for (int i = 1; i < dataC4; ++i) {
            out = vdivq_f32(in, vaddq_f32(one,out));
            vst1q_f32(dst ,out);
            dst += 4;
            src += 4;
            out = vld1q_f32(dst);
            in = vld1q_f32(src);
        }
        out = vdivq_f32(in, vaddq_f32(one,out));
        vst1q_f32(dst, out);
        dst += 4;
        src += 4;
    }
    if (remain > 0) {
        float intmp[4] = {0};
        float atmp[4] = {0};
        ::memcpy(intmp, dst, remain * sizeof(float));
        ::memcpy(atmp, src, remain * sizeof(float));
        float32x4_t out = vld1q_f32(intmp);
        float32x4_t in = vld1q_f32(atmp);
        out = vdivq_f32(in, vaddq_f32(one, out));
        vst1q_f32(intmp, out);
        ::memcpy(dst, intmp, remain * sizeof(float));
    }
#else
    for (int i = 0; i < dataSize; ++i) {
        dst[i] = src[i] / (1.0f + dst[i]);
    }
#endif
}


#ifdef MNN_LOW_MEMORY
static void generalIm2col(float* destOrigin, float const** sourceGroup, const int32_t* info, const int32_t* el, int LP, int pack) {
    // LP >= pack
    int number = info[0];
    int eReal = info[1];
    int eDest = info[2];
    int offset = info[3];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2];
        int lOffset = el[4 * n + 3];
        int lC = lOffset / LP;
        int lR = lOffset % LP;
        auto dest = destOrigin + eOffset * LP + lC * eDest * LP + lR;
        auto source = sourceGroup[n];

        for (int y=0; y<e; ++y) {
            auto yR = y % eDest;
            for (int x=0; x<l; ++x) {
                auto xR = x % pack;
                auto xC = x / pack;
                auto xOut = x / LP;
                auto xIn = x % LP;
                dest[xOut * eDest * LP + yR * LP + xIn] = source[xC * eReal * pack + y * pack * offset + xR];
            }
        }
    }
}
#endif // MNN_LOW_MEMORY

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
    gCoreFunction->MNNUnpackCUnit = MNNUnpackC4;
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
#ifdef MNN_LOW_MEMORY
    gCoreFunction->MNNAbsMax = MNNAbsMaxFP32;                      // abs max value for [icDiv4,plane,4] -> abs max:[plane]
    gCoreFunction->MNNDynamicQuant = MNNDynamicQuantFP32;          // symmetric 'batch' quant for [icDiv4,plane,4]
//     gCoreFunction->MNNAsyQuantFunc = MNNAsyQuantFunc;              // asymmetric 'batch' quant for [icDiv4,plane,4]
//     gCoreFunction->MNNAsyQuantInfo = MNNAsyQuantInfo_FP32;              // asymmetric quant/dequant scale&bias for [icDiv4,plane,4] -> scale&bias:[blockNum,plane]
    gCoreFunction->MNNQuantScale = MNNQuantScaleFP32;              // symmetric quant/dequant scale&bias for [icDiv4,plane,4] -> scale&bias:[plane]
    gCoreFunction->MNNGeneralIm2Col = generalIm2col;               // Im2Col based on float data -> output:[eU,kernelsize,lU,ep,lp]
//     gCoreFunction->MNNDynamicUpdateConvBiasScale = MNNDynamicUpdateConvBiasScale;
// #ifdef __aarch64__
//     if (gCoreFunction->supportSDot) {
//         gCoreFunction->MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Arm82;
//     }
//     if (gCoreFunction->supportI8mm) {
//         gCoreFunction->MNNGeneralIm2Col = MNNGeneralIm2col_Fp32Arm86;
//     }
// #endif
#endif
//     MNNCoreInt8FunctionInit();
//     MNNFunctionInit();
}

XPUCoreFunctions* MNNGetXPUCoreFunctions() {
    return gCoreFunction;
}

}; // namespace XPU
} // namespace MNN