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

} // namespace XPU
} // namespace MNN
