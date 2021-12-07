// Copyright (c) OpenMMLab. All rights reserved.

//
//#include "vulkan_device.h"
//
// namespace mmdeploy {
//
// VulkanPlatform::VulkanPlatform() {
//  vk::InstanceCreateInfo info{};
//  instance_ = vk::createInstance(info);
//  device_group_ = instance_.enumeratePhysicalDeviceGroups()[0];
//  ctx_.resize(device_group_.physicalDeviceCount);
//}
//
//
// int VulkanPlatform::CreateDevice(int idx) {
//  if (ctx_[idx].device) {
//    return M_SUCCESS;
//  }
//  try {
//    auto& physical_device = device_group_.physicalDevices[idx];
//    uint32_t queue_family = 0;
//    auto properties = physical_device.getQueueFamilyProperties();
//    for (size_t i = 0; i < properties.size(); ++i) {
//      auto& p = properties[i];
//      if (p.queueFlags & vk::QueueFlagBits::eCompute) {
//        fprintf(stderr, "%d\n", (int)p.queueCount);
//        queue_family = i;
//        break;
//      }
//    }
//    vk::DeviceQueueCreateInfo queue_info;
//    float queue_priority = 0.f;
//    queue_info.setQueueFamilyIndex(queue_family);
//    queue_info.setPQueuePriorities(&queue_priority);
//    queue_info.setQueueCount(1);
//
//    vk::DeviceCreateInfo device_info;
//    device_info.setQueueCreateInfoCount(1);
//    device_info.setPQueueCreateInfos(&queue_info);
//
//    auto device = physical_device.createDevice(device_info);
//
//    ctx_[idx].device = device;
//    ctx_[idx].queue = device.getQueue(queue_family, 0);
//  } catch (...) {
//    return Status(eFail);
//  }
//  return M_SUCCESS;
//}
//
// int VulkanPlatform::CreateBuffer(int32_t device_id, size_t size,
//                                 size_t alignment, Buffer* buffer) {
//  GetDevice(device_id);
//  return 0;
//}
//
// int VulkanPlatform::CreateStream(int32_t device_id, Stream* stream) {
//  *stream = Stream(std::make_shared<VulkanStream>(GetDevice(device_id),
//                                                  ctx_[device_id].queue));
//  return 0;
//}
//
// int VulkanPlatform::CreateEvent(int32_t device_id, Event* event) { return 0; }
//
// int VulkanPlatform::Copy(const void* host_ptr, Buffer dst, size_t size,
//                         size_t dst_offset, Stream stream) {
//  return 0;
//}
//
// int VulkanPlatform::Copy(Buffer src, void* host_ptr, size_t size,
//                         size_t src_offset, Stream stream) {
//  return 0;
//}
//
// int VulkanPlatform::Copy(Buffer src, Buffer dst, size_t size, size_t src_offset,
//                         size_t dst_offset, Stream stream) {
//  return 0;
//}
//
// VulkanBuffer::VulkanBuffer(DeviceType device, size_t size)
//    : BufferImpl(std::get<Device>(device)) {
//  vk::BufferCreateInfo info;
//  info.setSize(size);
//  info.setUsage(vk::BufferUsageFlagBits::eStorageBuffer);
//  std::get<vk::Device>(device).createBuffer(info);
//}
// int VulkanBuffer::Deinit() { return 0; }
// void* VulkanBuffer::GetNative(ErrorCode* ec) { return nullptr; }
// size_t VulkanBuffer::GetSize(ErrorCode* ec) { return 0; }
// void VulkanBuffer::MemoryHandleDestructor(MemoryHandle* memory) {}
//
//////////////////////////////////////////////////////////////////////////////////
///// OclPlatformRegisterer
//
// class VulkanPlatformRegisterer {
// public:
//  VulkanPlatformRegisterer() {
//    gPlatformRegistry().Register(
//        [] { return std::make_shared<VulkanPlatform>(); });
//  }
//};
//
// VulkanPlatformRegisterer g_vulkan_platform_registerer;
//
// VulkanStream::VulkanStream(DeviceType device, vk::Queue queue)
//  : StreamImpl(std::get<Device>(device)), queue_(queue) {}
//
//}  // namespace mmdeploy
