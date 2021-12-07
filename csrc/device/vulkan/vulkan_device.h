// Copyright (c) OpenMMLab. All rights reserved.

#ifndef MMDEPLOY_DEVICE_VULKAN_DEVICE_H_
#define MMDEPLOY_DEVICE_VULKAN_DEVICE_H_

#include "src/core/device/device_impl.h"
#include "vulkan/vulkan.hpp"

namespace mmdeploy {

using DeviceType = std::tuple<Device, vk::Device>;

class VulkanPlatform : public PlatformImpl {
 public:
  explicit VulkanPlatform();

  const char* GetPlatformName() const noexcept override { return "vulkan"; }

  int CreateBuffer(int32_t device_id, size_t size, size_t alignment, Buffer* buffer) override;

  int CreateStream(int32_t device_id, Stream* stream) override;

  int CreateEvent(int32_t device_id, Event* event) override;

  int Copy(const void* host_ptr, Buffer dst, size_t size, size_t dst_offset,
           Stream stream) override;

  int Copy(Buffer src, void* host_ptr, size_t size, size_t src_offset, Stream stream) override;

  int Copy(Buffer src, Buffer dst, size_t size, size_t src_offset, size_t dst_offset,
           Stream stream) override;

  DeviceType GetDevice(int idx) {
    CreateDevice(idx);
    return std::make_tuple(Device(platform_id_, idx), ctx_[idx].device);
  }

 private:
  int CreateDevice(int idx);

  struct Context {
    vk::Device device;
    vk::Queue queue;
  };

  vk::Instance instance_;
  vk::PhysicalDeviceGroupProperties device_group_;
  std::vector<Context> ctx_;
};

class VulkanMemory {
 public:
  explicit VulkanMemory(const vk::MemoryRequirements& req) {
    vk::UniqueHandle<vk::DeviceMemory, vk::DispatchLoaderStatic> memory;
    vk::MemoryAllocateInfo info;
    VkMemoryAllocateInfo memory = device_.allocateMemoryUnique()
  }
  ~VulkanMemory() { device_.free(memory_); }

 private:
  vk::Device device_;
  vk::DeviceMemory memory_;
};

class VulkanBuffer : public BufferImpl {
 public:
  explicit VulkanBuffer(DeviceType device, size_t size);

  int Init(size_t size, size_t alignment) override { return 0; }

  int Deinit() override;

  void* GetNative(ErrorCode* ec) override;

  size_t GetSize(ErrorCode* ec) override;

  vk::Buffer& get() { return *reinterpret_cast<vk::Buffer*>(&memory_->handle); }

 private:
  static void MemoryHandleDestructor(MemoryHandle* memory);

 private:
  std::shared_ptr<MemoryHandle> memory_;
  size_t size_{0};
};

class VulkanStream : public StreamImpl {
 public:
  explicit VulkanStream(DeviceType device, vk::Queue queue);

  ~VulkanStream() override;

  int DependsOn(Event& event) override;

  int Query() override;

  int Wait() override;

  int Submit(Kernel& kernel) override;

  void* GetNative(ErrorCode* ec) override;

  vk::Queue& get() { return queue_; }

 private:
  vk::Queue queue_;
};

class VulkanEvent : public EventImpl {
 public:
  explicit VulkanEvent(DeviceType device);

  ~VulkanEvent() override;

  int Query() override;

  int Record(Stream& stream) override;

  int Wait() override;

  void* GetNative(ErrorCode* ec) override;

 private:
  vk::Event event_;
};

}  // namespace mmdeploy

#endif  // MMDEPLOY_DEVICE_VULKAN_DEVICE_H_
