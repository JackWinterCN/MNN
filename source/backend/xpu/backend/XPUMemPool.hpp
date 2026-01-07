//
//  BufferPool.hpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#ifndef XPU_MEM_POOL_HPP
#define XPU_MEM_POOL_HPP

#include <map>
#include <memory>
#include <set>
#include <vector>
#include <core/Backend.hpp>
#include "core/NonCopyable.hpp"

namespace MNN {
namespace XPU {

struct XPUMemNode {
  XPUMemNode() {};
  ~XPUMemNode() {
    MNN_PRINT("[XPU] ~XPUMemNode(): 0x%lx size: %lu\n", physical_addr, size);
    if (physical_addr) {
      delete[] (int8_t *)physical_addr;
    }
  }
  size_t size{0};
  uint64_t physical_addr{0};
};

class XPUMemPool : public NonCopyable {
public:
  // XPUMemPool(cl::Context& context, cl_mem_flags flags) : mContext(context) {
  //     mFlag = flags;
  // }
  XPUMemPool() {}
  std::shared_ptr<XPUMemNode> alloc(size_t size, bool separate = false);
  void recycle(std::shared_ptr<XPUMemNode> node, bool release = false);
  void clear();
  void releaseFreeList();
  size_t totalSize() { return mTotalSize; }

private:
  std::set<std::shared_ptr<XPUMemNode>> mAllBuffer;
  std::multimap<size_t, std::shared_ptr<XPUMemNode>> mFreeList;
  // cl::Context& mContext;
  // cl_mem_flags mFlag;
  size_t mTotalSize = 0;
};

class XPUDeviceMemObj : public Backend::MemObj {
public:
  XPUDeviceMemObj(std::shared_ptr<XPUMemNode> node, XPUMemPool *bufferPool) {
    mNode = node;
    mBufferPool = bufferPool;
  }
  virtual ~XPUDeviceMemObj() { mBufferPool->recycle(mNode); }

private:
  std::shared_ptr<XPUMemNode> mNode;
  XPUMemPool *mBufferPool;
};

} // namespace XPU
} // namespace MNN

#endif /* XPU_MEM_POOL_HPP */
