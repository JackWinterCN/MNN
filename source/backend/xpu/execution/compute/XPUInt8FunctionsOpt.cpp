//
//  Int8FunctionsOpt.cpp
//  MNN
//
//  Created by MNN on 2018/08/15.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "XPUInt8FunctionsOpt.hpp"
#include <math.h>
#include <cstring> // for memset
#include "core/Macro.h"
#include "core/CommonCompute.hpp"
#include "math/Vec.hpp"
#ifdef MNN_USE_NEON
#include <arm_neon.h>
#endif

namespace MNN {
namespace XPU{

static void MNNGemmInt8AddBiasScale_16x4_Unit(int8_t* dst, const int8_t* src, const int8_t* weight, size_t src_depth_quad, size_t dst_step,
                                              size_t dst_depth_quad, const QuanPostTreatParameters* post, size_t realCount) {
    const int bytes = ((post->useInt8 == 1) ? 1 : 4);
    float fp32min = 0, fp32max = 0;
    int weight_step_Z = src_depth_quad * (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT) + 4 * 2 * GEMM_INT8_UNIT;
    int weight_step_Y = (GEMM_INT8_UNIT * GEMM_INT8_SRC_UNIT);
    
    if (0 == post->useInt8 && post->fp32minmax) {
        fp32min = (post->fp32minmax)[0];
        fp32max = (post->fp32minmax)[1];
    }

    float* biasPtr = (float*)post->biasFloat;
    auto accumbuff = post->accumBuffer;
    auto blockNum = post->blockNum;
    
    for (int dz = 0; dz < dst_depth_quad; ++dz) {
        auto dst_z              = dst + dz * dst_step;
        auto accum_z            = accumbuff;
        for (int bk = 0; bk < blockNum; ++bk) {
            // block's weight&scale&bias
            const auto weight_dz = weight + dz * blockNum * weight_step_Z + bk *  weight_step_Z;
            const float* scale_dz = reinterpret_cast<const float*>(weight_dz + src_depth_quad * weight_step_Y);
            const auto weightBias_dz = scale_dz + GEMM_INT8_UNIT;
            const auto bias_dz      = biasPtr + dz * GEMM_INT8_UNIT;

            const auto srcSumPtr = post->srcKernelSum + bk * realCount;
            
            for (int w = 0; w < realCount; ++w) {
                const auto src_x   = src + bk * src_depth_quad * GEMM_INT8_SRC_UNIT * realCount + w * GEMM_INT8_SRC_UNIT;
                auto dst_x         = dst_z + w * GEMM_INT8_UNIT * bytes;
                auto accum_x       = accum_z + w * GEMM_INT8_UNIT;
                int32_t dstTemp[4] = {0, 0, 0, 0};

                for (int sz = 0; sz < src_depth_quad; ++sz) {
                    const auto weight_sz = weight_dz + weight_step_Y * sz;
                    const auto src_z     = src_x + sz * realCount * GEMM_INT8_SRC_UNIT;

                    for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                        const auto weight_j = weight_sz + j * GEMM_INT8_SRC_UNIT;
                        for (int i = 0; i < GEMM_INT8_SRC_UNIT; ++i) {
                            dstTemp[j] += (int32_t)src_z[i] * (int32_t)weight_j[i];
                        }
                    }
                }

                for (int j = 0; j < GEMM_INT8_UNIT; ++j) {
                    float value = dstTemp[j] * scale_dz[j] + srcSumPtr[w] * weightBias_dz[j];
                    if (post->inputScale) {
                        value = dstTemp[j] * scale_dz[j] * (post->inputScale + bk * realCount)[w] + srcSumPtr[w] * weightBias_dz[j];
                    }
                    if (post->inputBias) {
                        auto weightKernelSum = post->weightKernelSum + dz * (blockNum * GEMM_INT8_UNIT) + bk * GEMM_INT8_UNIT;
                        value += ((post->inputBias + bk * realCount)[w] * weightKernelSum[j]);
                    }
                    if (post->useInt8 == 0) {
                        if (bk > 0) {
                            float dstv = ((float*)accum_x)[j];
                            value += dstv;
                        }
                        if (bk == blockNum - 1) {
                            if (biasPtr) {
                                value += bias_dz[j];
                            }
                            if (post->fp32minmax) {
                                value = std::min(std::max(fp32min, value), fp32max);
                            }
                            ((float*)dst_x)[j] = value;             
                        } else {
                            ((float*)accum_x)[j] = value;
                        }
                        
                    } else {
                        value += bias_dz[j];
                        value       = ALIMAX(value, post->minValue);
                        value       = ALIMIN(value, post->maxValue);
                        dst_x[j] = static_cast<int8_t>(roundf(value));
                    }
                }
            }
        }
    }
}


template<int EP, int LP, int HP>
static void _ArmBasicMNNPackC4ForMatMul_A(int8_t* destOrigin, int8_t const** sourceGroup, const int32_t* info, const int32_t* el) {
    int number = info[0];
    int eReal = info[1];
    int eOutsideStride = info[2] / sizeof(float);
    int eDest = EP;
    int offset = info[3];
    const int LUNIT = LP / sizeof(float);
    int realDstCount = info[4];
    for (int n=0; n<number; ++n) {
        int e = el[4 * n + 0];       // to fill
        int l = el[4 * n + 1];
        int eOffset = el[4 * n + 2]; // have filled
        int lOffset = el[4 * n + 3];
        int lC = lOffset / LP;
        int lR = lOffset % LP;
        int eC = eOffset / eDest;
        int eR = eOffset % eDest;
        int eS = eDest - eR;
//        printf("e=%d, eC=%d, lC=%d, eR=%d, lR=%d\n", e, eC, lC, eR, lR);
        bool lastBag = false;
        int eOutsideStride4LastBag = eOutsideStride;
        if (realDstCount % EP > 0) {
            int jobsE = realDstCount - eOffset - e;
            if (jobsE == 0 || (jobsE < (realDstCount % EP))) {
                lastBag = true;
            }
        }
        auto dest = (int32_t*)(destOrigin + lC * eDest * LP + lR + eC * info[2] + eR * LP);
        auto source = (int32_t*)sourceGroup[n];
        int lRemain = l / 4;
        int lR4 = lR / 4;
        int lS = LUNIT - lR4;
        
        if (lastBag && e + eR < EP) {
            int elast = ALIMAX(eR + e, realDstCount % EP);
            dest = (int32_t*)(destOrigin + lC * elast * LP + lR + eC * info[2] + eR * LP);
        }
        // Step for start
        int offsetLC = lC * LUNIT + lR / 4;

        if (lR4 > 0) {
            int step = ALIMIN(lS, lRemain);
            for (int x=0; x<step; ++x) {
                int eRemain = e;
                auto d = dest + x;
                auto s = source + x * eReal;
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR * LUNIT);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d += (eOutsideStride4LastBag - eR * LUNIT + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s += eS * offset;
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d+= (eOutsideStride4LastBag + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s+= eStep * offset;
                }
                offsetLC++;
            }
            lRemain -= step;
            if (lastBag && e + eR < EP) {
                int eFill = ALIMAX(realDstCount % EP, e + eR);
                int nextLP = (eFill * LP - lR) / sizeof(int32_t);
                dest += nextLP;
            } else {
                int nextLP = (eDest * LP - lR) / sizeof(int32_t);
                dest += nextLP;
            }
            source += eReal * step;
        }
        
        while (lRemain > 0) {
            int step = ALIMIN(lRemain, LUNIT);
            for (int x=0; x<step; ++x) {
                int eRemain = e;
                auto d = dest + x;
                auto s = source + x * eReal;
                if (eR > 0) {
                    int eStep = ALIMIN(eRemain, eS);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag ||eRemain >= EP) {
                        d += (eOutsideStride - eR * LUNIT);
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d += (eOutsideStride4LastBag - eR * LUNIT + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s += eS * offset;
                }
                while (eRemain > 0) {
                    int eStep = ALIMIN(eDest, eRemain);
                    for (int yi=0; yi<eStep; ++yi) {
                        d[yi * LUNIT] = s[yi * offset];
                    }
                    eRemain-=eStep;
                    if (!lastBag || eRemain >= EP) {
                        d+= eOutsideStride;
                    } else {
                        int eFill = ALIMAX(eRemain, realDstCount % EP); // maybe padding>0
                        eOutsideStride4LastBag = eOutsideStride - ((offsetLC / LUNIT) * EP * LUNIT);
                        d+= (eOutsideStride4LastBag + (offsetLC / LUNIT) * eFill * LUNIT);
                    }
                    s+= eStep * offset;
                }
                offsetLC++;
            }
            
            lRemain -= step;
            if (lastBag && e + eR < EP) {
                int efill = ALIMAX(e + eR, realDstCount % EP);
                dest += efill * LUNIT;
            } else {
                dest += eDest * LUNIT;
            }
            source += eReal * step;
        }
    }
}

static void MNNGetGemmUnit(int* UNIT, int* SRC_UNIT, int* DST_XUNIT) {
    *UNIT = GEMM_INT8_UNIT;
    *SRC_UNIT = GEMM_INT8_SRC_UNIT;
    *DST_XUNIT = GEMM_INT8_DST_XUNIT;
}


static XPUCoreInt8Functions* gCoreFunc = nullptr;

void MNNXPUCoreInt8FunctionInit() {
    /* CoreInt8Functions without sdot */
    gCoreFunc = new XPUCoreInt8Functions;

    // MatMul
    gCoreFunc->Int8GemmKernel = MNNGemmInt8AddBiasScale_16x4_Unit;

    gCoreFunc->MNNGetGemmUnit = MNNGetGemmUnit;

    // Im2Col
    gCoreFunc->MNNPackC4Int8ForMatMul_A = _ArmBasicMNNPackC4ForMatMul_A<GEMM_INT8_DST_XUNIT, GEMM_INT8_SRC_UNIT, GEMM_INT8_UNIT>;

}

XPUCoreInt8Functions* MNNXPUGetInt8CoreFunctions() {
    return gCoreFunc;
}


} // namespace XPU
} // namespace MNN
