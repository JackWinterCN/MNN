//
//  BufferPool.cpp
//  MNN
//
//  Created by MNN on 2019/02/28.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "backend/xpu/backend/XPUMemPool.hpp"
namespace MNN {
namespace XPU {

std::shared_ptr<XPUMemNode> XPUMemPool::alloc(size_t size, bool separate) {
  if (!separate) {
    // FIX: this reuse strategy is not correct, it may destroy the used memory
    // auto iter = mFreeList.lower_bound(size);
    // if (iter != mFreeList.end()) {
    //   auto node = iter->second;
    //   mFreeList.erase(iter);
    //   MNN_PRINT("[XPU] XPUMemPool reuse addr: 0x%lx size: %lu\n",
    //             node->physical_addr, node->size);
    //   return node;
    // }
  
    // TODO: this branch may lead to dangling pointer, because the node in
    // FreeLists may be used by other operator, and the following code will free
    // the memory of the node, which may cause dangling pointer.
    // else if (mFreeList.size() != 0) {
    //   auto maxIter = mFreeList.rbegin();
    //   auto node = maxIter->second;
    //   // free old memory
    //   if (node->physical_addr) {
    //     MNN_PRINT("[XPU] XPUMemPool free addr: 0x%lx size: %lu\n",
    //               node->physical_addr, node->size);
    //     delete[] (int8_t *)node->physical_addr;
    //     node->physical_addr = 0;
    //     mTotalSize -= node->size;
    //     node->size = 0;
    //   }
    //   // allocate new memory
    //   auto p = new int8_t[size];
    //   if (nullptr == p) {
    //     MNN_ERROR("Alloc Buffer %lu error\n", size);
    //     return nullptr;
    //   }
    //   node->size = size;
    //   node->physical_addr = (uint64_t)(p);
    //   mTotalSize += size;
    //   mFreeList.erase(std::prev(mFreeList.end()));
    //   MNN_PRINT("[XPU] XPUMemPool alloc addr: 0x%lx size: %lu\n",
    //             node->physical_addr, size);
    //   return node;
    // }
  }
  std::shared_ptr<XPUMemNode> node(new XPUMemNode);
  auto p = new int8_t[size];
  if (nullptr == p) {
    MNN_ERROR("Alloc Buffer %lu error\n", size);
    return nullptr;
  }
  node->size = size;
  node->physical_addr = (uint64_t)(p);
  mTotalSize += size;
  mAllBuffer.insert(node);
  MNN_PRINT("[XPU] XPUMemPool alloc addr: 0x%lx size: %lu\n",
            node->physical_addr, size);
  return node;
}

void XPUMemPool::recycle(std::shared_ptr<XPUMemNode> node, bool release) {
  auto iter = mAllBuffer.find(node);
  if (iter == mAllBuffer.end()) {
    MNN_ERROR("Error for recycle buffer\n");
    return;
  }
  if (release) {
    MNN_PRINT("[XPU] XPUMemPool recycle RELEASE addr: 0x%lx size: %lu\n",
              node->physical_addr, node->size);
    mAllBuffer.erase(node);
    return;
  }
  MNN_PRINT("[XPU] XPUMemPool recycle addr: 0x%lx size: %lu\n",
            node->physical_addr, node->size);
  mFreeList.insert(std::make_pair(node->size, node));
}

void XPUMemPool::clear() {
  MNN_PRINT("[XPU] XPUMemPool clear\n");
  mFreeList.clear();
  mAllBuffer.clear();
  mTotalSize = 0;
}

void XPUMemPool::releaseFreeList() {
  for (auto mf : mFreeList) {
    auto iter = mAllBuffer.find(mf.second);
    if (iter != mAllBuffer.end()) {
      mAllBuffer.erase(iter);
    }
  }
  mFreeList.clear();
}

} // namespace XPU
} // namespace MNN
