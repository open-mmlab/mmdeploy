// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_SRC_CORE_DEVICE_IMPL_H_
#define MMDEPLOY_SRC_CORE_DEVICE_IMPL_H_

#include "mmdeploy/core/device.h"

namespace mmdeploy::framework {

using std::shared_ptr;

using PlatformImplPtr = shared_ptr<PlatformImpl>;
using AllocatorImplPtr = shared_ptr<AllocatorImpl>;
using BufferImplPtr = shared_ptr<BufferImpl>;
using StreamImplPtr = shared_ptr<StreamImpl>;
using EventImplPtr = shared_ptr<EventImpl>;

class PlatformImpl {
 public:
  PlatformImpl() : platform_id_(-1) {}

  virtual ~PlatformImpl() = default;

  virtual const char* GetPlatformName() const noexcept = 0;

  virtual int GetPlatformId() const noexcept { return platform_id_; }

  virtual void SetPlatformId(int id) { platform_id_ = id; }

  virtual Result<void> BindDevice(Device device, Device* prev) = 0;

  virtual shared_ptr<BufferImpl> CreateBuffer(Device device) = 0;

  virtual shared_ptr<StreamImpl> CreateStream(Device device) = 0;

  virtual shared_ptr<EventImpl> CreateEvent(Device device) = 0;

  virtual Result<void> Copy(const void* host_ptr, Buffer dst, size_t size, size_t dst_offset,
                            Stream stream) = 0;

  virtual Result<void> Copy(Buffer src, void* host_ptr, size_t size, size_t src_offset,
                            Stream stream) = 0;

  virtual Result<void> Copy(Buffer src, Buffer dst, size_t size, size_t src_offset,
                            size_t dst_offset, Stream stream) = 0;

  virtual Result<Stream> GetDefaultStream(int32_t device_id) = 0;

 protected:
  int platform_id_;
};

class AllocatorImpl {
 public:
  struct Block {
    explicit Block(void* _handle = nullptr, size_t _size = 0) : handle(_handle), size(_size) {}
    void* handle;
    size_t size;
  };
  virtual ~AllocatorImpl() = default;
  virtual Block Allocate(size_t size) noexcept = 0;
  virtual void Deallocate(Block& block) noexcept = 0;
  virtual bool Owns(const Block& block) const noexcept = 0;
  virtual const char* Name() const noexcept { return ""; }
  //  virtual Device device() const noexcept = 0;
};

// create, destroy, sub, MakeAvailableOnDevice, FromHost, fill, copy, map, unmap
class BufferImpl {
 public:
  explicit BufferImpl(Device device) : device_(device) {}

  virtual ~BufferImpl() = default;

  virtual Result<void> Init(size_t size, Allocator allocator, size_t alignment, uint64_t flags) = 0;

  virtual Result<void> Init(size_t size, std::shared_ptr<void> native, uint64_t flags) = 0;

  virtual Result<shared_ptr<BufferImpl>> SubBuffer(size_t offset, size_t size, uint64_t flags) = 0;

  virtual size_t GetSize(ErrorCode* ec) = 0;

  virtual Allocator GetAllocator() const = 0;

  virtual void* GetNative(ErrorCode* ec) = 0;

  Device GetDevice() const noexcept { return device_; }

 protected:
  Device device_;
};

class StreamImpl {
 public:
  explicit StreamImpl(Device device) : device_(device) {}

  virtual ~StreamImpl() = default;

  virtual Result<void> Init(uint64_t flags) = 0;

  virtual Result<void> Init(std::shared_ptr<void> native, uint64_t flags) = 0;

  virtual Result<void> Query() = 0;

  virtual Result<void> Wait() = 0;

  virtual Result<void> Submit(Kernel& kernel) = 0;

  virtual Result<void> DependsOn(Event& event) = 0;

  virtual void* GetNative(ErrorCode* ec) = 0;

  Device GetDevice() const noexcept { return device_; }

 protected:
  Device device_;
};

class EventImpl {
 public:
  explicit EventImpl(Device device) : device_(device) {}

  virtual ~EventImpl() = default;

  virtual Result<void> Init(uint64_t flags) = 0;

  virtual Result<void> Init(std::shared_ptr<void> native, uint64_t flags) = 0;

  virtual Result<void> Query() = 0;

  virtual Result<void> Record(Stream& st) = 0;

  virtual Result<void> Wait() = 0;

  virtual void* GetNative(ErrorCode* ec) = 0;

  Device GetDevice() const noexcept { return device_; }

 protected:
  Device device_;
};

class KernelWrapper {
 public:
  virtual ~KernelWrapper() = default;
  virtual int Invoke(const std::vector<void*>& args) = 0;
};

class KernelImpl {
 public:
  explicit KernelImpl(Device device) : device_(device) {}

  virtual ~KernelImpl() = default;

  Device GetDevice() const noexcept { return device_; }

  virtual void* GetNative(ErrorCode* ec) = 0;

 protected:
  Device device_;
};

struct Access {
  template <typename T, typename Obj>
  static T& get(const Obj& obj) {
    return static_cast<T&>(*obj.impl_);
  }

  template <typename Obj>
  static auto& get_impl(const Obj& obj) {
    return obj.impl_;
  }

  template <typename T, typename... Args>
  static T create(Args&&... args) {
    return T(std::forward<Args>(args)...);
  }
};

inline PlatformImpl* GetPlatformImpl(const Device& device) {
  return gPlatformRegistry().GetPlatformImpl(device);
}

}  // namespace mmdeploy::framework

#endif
