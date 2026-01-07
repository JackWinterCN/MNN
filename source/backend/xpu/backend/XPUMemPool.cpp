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
    auto iter = mFreeList.lower_bound(size);
    if (iter != mFreeList.end()) {
      auto node = iter->second;
      mFreeList.erase(iter);
      return node;
    } else if (mFreeList.size() != 0) {
      auto maxIter = mFreeList.rbegin();
      auto node = maxIter->second;
      mTotalSize += size - node.get()->size;
      node.get()->size = size;
      auto p = new int8_t[size];
      if (nullptr == p) {
        MNN_ERROR("Alloc Buffer %lu error\n", size);
        return nullptr;
      }
      node.get()->physical_addr = (uint64_t)(p);
      mFreeList.erase(std::prev(mFreeList.end()));
      MNN_PRINT("[XPU] XPUMemPool alloc addr: 0x%lx size: %lu\n",
                node.get()->physical_addr, size);
      return node;
    }
  }
  std::shared_ptr<XPUMemNode> node(new XPUMemNode);
  mTotalSize += size;
  node->size = size;
  auto p = new int8_t[size];
  if (nullptr == p) {
    MNN_ERROR("Alloc Buffer %lu error\n", size);
    return nullptr;
  }
  node.get()->physical_addr = (uint64_t)(p);
  mAllBuffer.insert(node);
  MNN_PRINT("[XPU] XPUMemPool alloc addr: 0x%lx size: %lu\n",
            node.get()->physical_addr, size);
  return node;
}

void XPUMemPool::recycle(std::shared_ptr<XPUMemNode> node, bool release) {
  auto iter = mAllBuffer.find(node);
  if (iter == mAllBuffer.end()) {
    MNN_ERROR("Error for recycle buffer\n");
    return;
  }
  if (release) {
    mAllBuffer.erase(node);
    return;
  }
  mFreeList.insert(std::make_pair(node.get()->size, node));
}

void XPUMemPool::clear() {
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
