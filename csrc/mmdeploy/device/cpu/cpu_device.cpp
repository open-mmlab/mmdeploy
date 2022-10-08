// Copyright (c) OpenMMLab. All rights reserved.
#include "cpu_device.h"

#include <cassert>
#include <cstdlib>
#include <cstring>

namespace mmdeploy {

class CpuHostMemory : public NonCopyable {
 public:
  CpuHostMemory() : size_(), data_(), owned_data_{false} {}
  Result<void> Init(size_t size, size_t alignment) {
    if (alignment != 1) {
      return Status(eNotSupported);
    }
    data_ = std::malloc(size);
    if (!data_) {
      return Status(eOutOfMemory);
    }
    size_ = size;
    owned_data_ = true;
    return success();
  }
  Result<void> Init(size_t size, std::shared_ptr<void> data) {
    size_ = size;
    external_ = std::move(data);
    data_ = external_.get();
    owned_data_ = false;
    return success();
  }
  Result<void> Init(size_t size, void* data) {
    size_ = size;
    data_ = data;
    owned_data_ = false;
    return success();
  }
  ~CpuHostMemory() {
    if (data_) {
      if (owned_data_) {
        std::free(data_);
        owned_data_ = false;
      }
      data_ = nullptr;
    }
    external_.reset();
    size_ = 0;
  }
  size_t size() const { return size_; }
  void* data() const { return data_; }

 private:
  size_t size_;
  void* data_;
  bool owned_data_;
  std::shared_ptr<void> external_;
};

////////////////////////////////////////////////////////////////////////////////
/// CpuPlatformImpl

shared_ptr<BufferImpl> CpuPlatformImpl::CreateBuffer(Device device) {
  return std::make_shared<CpuBufferImpl>(device);
}

shared_ptr<StreamImpl> CpuPlatformImpl::CreateStream(Device device) {
  return std::make_shared<CpuStreamImpl>(device);
}

shared_ptr<EventImpl> CpuPlatformImpl::CreateEvent(Device device) {
  return std::make_shared<CpuEventImpl>(device);
}

int CpuPlatformImpl::GetPlatformId() const noexcept { return 0; }

const char* CpuPlatformImpl::GetPlatformName() const noexcept { return "cpu"; }

bool CpuPlatformImpl::CheckCopyParam(size_t src_size, size_t dst_size, size_t src_offset,
                                     size_t dst_offset, size_t copy_size) {
  if (src_offset + copy_size > src_size) {
    return false;
  }
  if (dst_offset + copy_size > dst_size) {
    return false;
  }
  return true;
}

inline void* OffsetPtr(void* ptr, size_t offset) {
  return static_cast<void*>(static_cast<uint8_t*>(ptr) + offset);
}

inline const void* OffsetPtr(const void* ptr, size_t offset) {
  return static_cast<const void*>(static_cast<const uint8_t*>(ptr) + offset);
}

Result<void> CpuPlatformImpl::CopyImpl(const void* src, void* dst, size_t src_size, size_t dst_size,
                                       size_t src_offset, size_t dst_offset, size_t size,
                                       Stream st) {
  if (!CheckCopyParam(src_size, dst_size, src_offset, dst_offset, size)) {
    return Status(eInvalidArgument);
  }
  auto task = [=] { std::memcpy(OffsetPtr(dst, dst_offset), OffsetPtr(src, src_offset), size); };
  if (!st) {
    task();
    return success();
  }
  if (st.GetDevice().platform_id() != 0) {
    return Status(eInvalidArgument);
  }
  auto cpu_stream = static_cast<CpuStreamImpl*>(st.GetNative());
  if (!cpu_stream) {
    return Status(eInvalidArgument);
  }
  return cpu_stream->Enqueue(std::move(task));
}

Result<void> CpuPlatformImpl::Copy(const void* host_ptr, Buffer dst, size_t size, size_t dst_offset,
                                   Stream stream) {
  auto dst_ptr = dst.GetNative();
  if (!dst_ptr) {
    return Status(eInvalidArgument);
  }
  if (dst.GetDevice().platform_id() != 0) {
    return Status(eInvalidArgument);
  }
  return CopyImpl(host_ptr, dst_ptr, size, dst.GetSize(), 0, dst_offset, size, stream);
}

Result<void> CpuPlatformImpl::Copy(Buffer src, void* host_ptr, size_t size, size_t src_offset,
                                   Stream stream) {
  auto src_ptr = src.GetNative();
  if (!src_ptr) {
    return Status(eInvalidArgument);
  }
  if (src.GetDevice().platform_id() != 0) {
    return Status(eInvalidArgument);
  }
  return CopyImpl(src_ptr, host_ptr, src.GetSize(), size, src_offset, 0, size, stream);
}
Result<void> CpuPlatformImpl::Copy(Buffer src, Buffer dst, size_t size, size_t src_offset,
                                   size_t dst_offset, Stream stream) {
  auto src_ptr = src.GetNative();
  auto dst_ptr = dst.GetNative();
  if (!src_ptr || !dst_ptr) {
    return Status(eInvalidArgument);
  }
  auto device = src.GetDevice();
  if (device.platform_id() != 0 || device.platform_id() != dst.GetDevice().platform_id()) {
    return Status(eInvalidArgument);
  }
  return CopyImpl(src_ptr, dst_ptr, src.GetSize(), dst.GetSize(), src_offset, dst_offset, size,
                  stream);
}

Result<Stream> CpuPlatformImpl::GetDefaultStream(int32_t device_id) {
  try {
    std::call_once(init_flag_, [&] { default_stream_ = Stream(GetDevice(device_id)); });
    return default_stream_;
  } catch (...) {
    return Status(eFail);
  }
}

////////////////////////////////////////////////////////////////////////////////
/// CpuBufferImpl

CpuBufferImpl::CpuBufferImpl(Device device) : BufferImpl(device) {}

void* CpuBufferImpl::GetNative(ErrorCode* ec) {
  if (!memory_) {
    if (ec) *ec = eInvalidArgument;
    return nullptr;
  }
  if (ec) *ec = ErrorCode::eSuccess;
  return OffsetPtr(memory_->data(), offset_);
}

Allocator CpuBufferImpl::GetAllocator() const { return {}; }

size_t CpuBufferImpl::GetSize(ErrorCode* ec) {
  if (!memory_) {
    if (ec) *ec = eInvalidArgument;
    return 0;
  }
  if (ec) *ec = ErrorCode::eSuccess;
  return size_;
}

// int CpuBufferImpl::Fill(uint8_t pattern, size_t size, size_t offset,
//                         Stream& st) {
//   if (!memory_ || !memory_->handle) {
//     return Status(eInvalidArgument);
//   }
//   if (offset + size >= size_) {
//     return Status(eInvalidArgument);
//   }
//   auto task = [=] {
//     auto data = OffsetPtr(memory_->handle, offset);
//     std::memset(data, pattern, size);
//   };
//   if (!st) {
//     task();
//     return M_SUCCESS;
//   }
//   if (st.GetDevice() != Device()) {
//     return Status(eInvalidArgument);
//   }
//   auto cpu_stream = static_cast<CpuStreamImpl*>(st.GetNative());
//   if (!cpu_stream) {
//     return Status(eInvalidArgument);
//   }
//   return cpu_stream->Enqueue(std::move(task));
// }

Result<void> CpuBufferImpl::Init(size_t size, Allocator allocator, size_t alignment,
                                 uint64_t flags) {
  assert(!allocator && "CPU device doesn't support allocators yet");
  memory_ = std::make_shared<CpuHostMemory>();
  OUTCOME_TRY(memory_->Init(size, alignment));
  size_ = size;
  return success();
}

Result<void> CpuBufferImpl::Init(size_t size, std::shared_ptr<void> native, uint64_t flags) {
  memory_ = std::make_shared<CpuHostMemory>();
  OUTCOME_TRY(memory_->Init(size, std::move(native)));
  size_ = size;
  return success();
}

Result<BufferImplPtr> CpuBufferImpl::SubBuffer(size_t offset, size_t size, uint64_t flags) {
  if (offset_ + offset + size > memory_->size()) {
    return Status(eInvalidArgument);
  }
  auto impl = std::make_shared<CpuBufferImpl>(device_);
  impl->memory_ = memory_;
  impl->offset_ = offset_ + offset;
  impl->size_ = size;
  return impl;
}

////////////////////////////////////////////////////////////////////////////////
/// CpuStreamImpl

CpuStreamImpl::CpuStreamImpl(Device device) : StreamImpl(device) {}

CpuStreamImpl::~CpuStreamImpl() {
  {
    std::lock_guard lock(mutex_);
    abort_ = true;
  }
  cv_.notify_one();
  thread_.join();
}

Result<void> CpuStreamImpl::Init(uint64_t flags) {
  thread_ = std::thread(&CpuStreamImpl::InternalThreadEntry, this);
  return success();
}

Result<void> CpuStreamImpl::Init(std::shared_ptr<void> native, uint64_t flags) {
  return Status(eNotSupported);
}

Result<void> CpuStreamImpl::Enqueue(Task task) {
  {
    std::lock_guard lock(mutex_);
    task_queue_.push(std::move(task));
  }
  cv_.notify_one();
  return success();
}

Result<void> CpuStreamImpl::DependsOn(Event& event) {
  return Enqueue([&] { event.Wait().value(); });
}

Result<void> CpuStreamImpl::Query() {
  std::lock_guard lock(mutex_);
  if (task_queue_.empty()) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> CpuStreamImpl::Wait() {
  {
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this] { return task_queue_.empty() || abort_; });
  }
  cv_.notify_one();
  return success();
}

Result<void> CpuStreamImpl::Submit(Kernel& kernel) {
  if (GetDevice() != kernel.GetDevice()) {
    return Status(eInvalidArgument);
  }
  auto task = static_cast<Task*>(kernel.GetNative());
  if (task) {
    OUTCOME_TRY(Enqueue(*task));
    return success();
  }
  return Status(eInvalidArgument);
}

void* CpuStreamImpl::GetNative(ErrorCode* ec) {
  if (ec) *ec = ErrorCode::eSuccess;
  return this;
}

void CpuStreamImpl::InternalThreadEntry() {
  while (true) {
    Task task;
    {
      std::unique_lock lock(mutex_);
      cv_.wait(lock, [this] { return !task_queue_.empty() || abort_; });
      if (abort_) {
        break;
      }
      task = std::move(task_queue_.front());
    }
    if (task) {
      task();
    }
    {
      std::lock_guard lock(mutex_);
      task_queue_.pop();
    }
    cv_.notify_one();
  }
}

////////////////////////////////////////////////////////////////////////////////
/// CpuEventImpl

CpuEventImpl::CpuEventImpl(Device device) : EventImpl(device) {}

Result<void> CpuEventImpl::Init(uint64_t flags) {
  Reset();
  return success();
};

Result<void> CpuEventImpl::Init(std::shared_ptr<void> native, uint64_t flags) {
  return Status(eNotSupported);
};

Result<void> CpuEventImpl::Query() {
  auto status = future_.wait_for(std::chrono::microseconds::zero());
  if (status == std::future_status::ready) {
    return success();
  } else {
    return Status(eNotReady);
  }
}

Result<void> CpuEventImpl::Record(Stream& stream) {
  if (stream.GetDevice() != device_) {
    return Status(eInvalidArgument);
  }
  auto cpu_stream = static_cast<CpuStreamImpl*>(stream.GetNative());
  if (!cpu_stream) return Status(eInvalidArgument);
  Reset();
  return cpu_stream->Enqueue([this] { promise_.set_value(); });
}

Result<void> CpuEventImpl::Wait() {
  future_.wait();
  return success();
};

void CpuEventImpl::Reset() {
  promise_ = std::promise<void>();
  future_ = promise_.get_future();
}

void* CpuEventImpl::GetNative(ErrorCode* ec) {
  if (ec) *ec = ErrorCode::eSuccess;
  return this;
}

////////////////////////////////////////////////////////////////////////////////

Kernel CreateCpuKernel(std::function<void()> task) {
  return Kernel(std::make_shared<CpuKernelImpl>(gCpuPlatform().GetDevice(0), std::move(task)));
}

////////////////////////////////////////////////////////////////////////////////
/// CpuPlatformRegisterer

CpuPlatformImpl& gCpuPlatform() {
  static Platform platform("cpu");
  return Access::get<CpuPlatformImpl>(platform);
}

class CpuPlatformRegisterer {
 public:
  CpuPlatformRegisterer() {
    gPlatformRegistry().Register([] { return std::make_shared<CpuPlatformImpl>(); });
  }
};

CpuPlatformRegisterer g_cpu_platform_registerer;

}  // namespace mmdeploy
