// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_DEVICE_OPENCL_DEVICE_H_
#define MMDEPLOY_DEVICE_OPENCL_DEVICE_H_

#include <mutex>

#include "CL/cl.hpp"
#include "core/device_impl.h"
#include "core/logger.h"

namespace mmdeploy {

namespace detail {

static inline cl::CommandQueue& Cast(cl_command_queue& queue) {
  return *reinterpret_cast<cl::CommandQueue*>(&queue);
}

static inline cl::Buffer& Cast(cl_mem& buffer) { return *reinterpret_cast<cl::Buffer*>(&buffer); }

static inline cl::Event& Cast(cl_event& event) { return *reinterpret_cast<cl::Event*>(&event); }

static inline cl::Context& Cast(cl_context& context) {
  return *reinterpret_cast<cl::Context*>(&context);
}

static inline cl::Platform& Cast(cl_platform_id& platform) {
  return *reinterpret_cast<cl::Platform*>(&platform);
}

static inline cl::Device& Cast(cl_device_id& device) {
  return *reinterpret_cast<cl::Device*>(&device);
}

}  // namespace detail

class OclPlatformImpl : public PlatformImpl {
 public:
  explicit OclPlatformImpl(cl::Platform platform);

  const char* GetPlatformName() const noexcept override { return "opencl"; }

  shared_ptr<BufferImpl> CreateBuffer(Device device) override;

  shared_ptr<StreamImpl> CreateStream(Device device) override;

  shared_ptr<EventImpl> CreateEvent(Device device) override;

  Result<void> Copy(const void* host_ptr, Buffer dst, size_t size, size_t dst_offset,
                    Stream stream) override;

  Result<void> Copy(Buffer src, void* host_ptr, size_t size, size_t src_offset,
                    Stream stream) override;

  Result<void> Copy(Buffer src, Buffer dst, size_t size, size_t src_offset, size_t dst_offset,
                    Stream stream) override;

  Result<Stream> GetDefaultStream(int32_t device_id) override;

  Device GetDevice(int device_id) { return Device(platform_id_, device_id); }

  cl::Device& GetNativeDevice(int device_id) { return devices_[device_id]; }

  cl::Context& GetContext() { return ctx_; }

 private:
  cl::Platform platform_;
  std::vector<cl::Device> devices_;
  std::vector<Stream> queues_;
  std::vector<std::unique_ptr<std::once_flag>> init_flag_;
  cl::Context ctx_;
};

OclPlatformImpl& gOclPlatform();

class OclDeviceMemory {
 public:
  OclDeviceMemory() : size_(), data_(), owned_data_(false) {}
  Result<void> Init(const cl::Context& ctx, size_t size, size_t alignment, uint64_t flags) {
    if (alignment != 1) {
      return Status(eNotSupported);
    }
    new (&data_) cl::Buffer(ctx, CL_MEM_READ_WRITE, size);
    owned_data_ = true;
    size_ = size;
    return success();
  }
  Result<void> Init(size_t size, shared_ptr<void> data, uint64_t flags) {
    external_ = std::move(data);
    data_ = static_cast<cl_mem>(external_.get());
    size_ = size;
    return success();
  }
  ~OclDeviceMemory() {
    if (owned_data_) {
      detail::Cast(data_).~Buffer();
      owned_data_ = false;
    }
    size_ = 0;
    data_ = cl_mem{};
    external_.reset();
  }
  size_t size() const { return size_; }
  cl_mem& data() { return data_; }

 private:
  size_t size_;
  cl_mem data_;
  bool owned_data_;
  shared_ptr<void> external_;
};

class OclBufferImpl : public BufferImpl {
 public:
  explicit OclBufferImpl(Device device);

  Result<void> Init(size_t size, Allocator allocator, size_t alignment, uint64_t flags) override;

  Result<void> Init(size_t size, std::shared_ptr<void> native, uint64_t flags) override;

  Result<BufferImplPtr> SubBuffer(size_t offset, size_t size, uint64_t flags) override {
    return Status(eNotSupported);
  }

  void* GetNative(ErrorCode* ec) override;

  size_t GetSize(ErrorCode* ec) override;

  cl::Buffer& buffer() { return detail::Cast(memory_->data()); }

 private:
  std::shared_ptr<OclDeviceMemory> memory_;
  size_t size_{0};
};

class OclStreamImpl : public StreamImpl {
 public:
  explicit OclStreamImpl(Device device);

  ~OclStreamImpl() override;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> DependsOn(Event& event) override;

  Result<void> Query() override;

  Result<void> Wait() override;

  Result<void> Submit(Kernel& kernel) override;

  void* GetNative(ErrorCode* ec) override;

  cl::CommandQueue& queue() { return detail::Cast(queue_); }

 private:
  cl_command_queue queue_;
  bool owned_queue_;
  std::shared_ptr<void> external_;
};

class OclEventImpl : public EventImpl {
 public:
  explicit OclEventImpl(Device device);

  ~OclEventImpl() override;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> Query() override;

  Result<void> Record(Stream& stream) override;

  Result<void> Wait() override;

  void* GetNative(ErrorCode* ec) override;

  cl::Event& event() { return detail::Cast(event_); }

 private:
  cl_event event_;
  bool owned_event_;
  std::shared_ptr<void> external_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_DEVICE_OPENCL_DEVICE_H_
