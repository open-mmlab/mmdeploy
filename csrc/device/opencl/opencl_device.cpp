// Copyright (c) OpenMMLab. All rights reserved.

#if ENABLE_OPENCL && 0

#include "opencl_device.h"

#include <iostream>
#include <mutex>

namespace mmdeploy {

////////////////////////////////////////////////////////////////////////////////
/// OclPlatformImpl

OclPlatformImpl::OclPlatformImpl(cl::Platform platform) : platform_(std::move(platform)) {
  platform_.getDevices(CL_DEVICE_TYPE_ALL, &devices_);
  queues_.resize(devices_.size());
  for (int i = 0; i < devices_.size(); ++i) {
    init_flag_.push_back(std::make_unique<std::once_flag>());
  }
  ctx_ = cl::Context(devices_);
}

shared_ptr<BufferImpl> OclPlatformImpl::CreateBuffer(Device device) {
  return std::make_shared<OclBufferImpl>(device);
}

shared_ptr<StreamImpl> OclPlatformImpl::CreateStream(Device device) {
  return std::make_shared<OclStreamImpl>(device);
}

shared_ptr<EventImpl> OclPlatformImpl::CreateEvent(Device device) {
  return std::make_shared<OclEventImpl>(device);
}

Result<void> OclPlatformImpl::Copy(const void* host_ptr, Buffer dst, size_t size, size_t dst_offset,
                                   Stream stream) {
  if (!dst || !stream) {
    return Status(eInvalidArgument);
  }
  auto device = dst.GetDevice();
  if (device.platform_id() != GetPlatformId()) {
    return Status(eInvalidArgument);
  }
  if (stream.GetDevice() != device) {
    return Status(eInvalidArgument);
  }

  auto& queue = Access::get<OclStreamImpl>(stream).queue();
  auto& to = Access::get<OclBufferImpl>(dst).buffer();
  auto status = queue.enqueueWriteBuffer(to, false, dst_offset, size, host_ptr);
  if (status == CL_SUCCESS) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> OclPlatformImpl::Copy(Buffer src, void* host_ptr, size_t size, size_t src_offset,
                                   Stream stream) {
  auto& queue = Access::get<OclStreamImpl>(stream).queue();
  auto& from = Access::get<OclBufferImpl>(src).buffer();

  auto status = queue.enqueueReadBuffer(from, false, src_offset, size, host_ptr);
  if (status) {
    fprintf(stderr, "status = %d\n", (int)status);
  }
  if (status == CL_SUCCESS) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> OclPlatformImpl::Copy(Buffer src, Buffer dst, size_t size, size_t src_offset,
                                   size_t dst_offset, Stream stream) {
  auto& queue = Access::get<OclStreamImpl>(stream).queue();
  auto& from = Access::get<OclBufferImpl>(src).buffer();
  auto& to = Access::get<OclBufferImpl>(dst).buffer();

  auto status = queue.enqueueCopyBuffer(from, to, src_offset, dst_offset, size);
  if (status) {
    fprintf(stderr, "status = %d\n", (int)status);
  }
  if (status == CL_SUCCESS) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<Stream> OclPlatformImpl::GetDefaultStream(int32_t device_id) {
  if (device_id >= queues_.size()) {
    return Status(eInvalidArgument);
  }
  try {
    std::call_once(*init_flag_[device_id],
                   [&] { queues_[device_id] = Stream(GetDevice((device_id))); });
    return queues_[device_id];
  } catch (...) {
    return Status(eFail);
  }
}

OclPlatformImpl& gOclPlatform() {
  static Platform platform("opencl");
  return Access::get<OclPlatformImpl>(platform);
}

////////////////////////////////////////////////////////////////////////////////
/// OclBufferImpl

OclBufferImpl::OclBufferImpl(Device device) : BufferImpl(device) {
  memory_ = std::make_shared<OclDeviceMemory>();
}

Result<void> OclBufferImpl::Init(size_t size, Allocator allocator, size_t alignment,
                                 uint64_t flags) {
  auto& ctx = gOclPlatform().GetContext();
  OUTCOME_TRY(memory_->Init(ctx, size, alignment, flags));
  size_ = size;
  return success();
}

Result<void> OclBufferImpl::Init(size_t size, std::shared_ptr<void> native, uint64_t flags) {
  OUTCOME_TRY(memory_->Init(size, std::move(native), flags));
  size_ = size;
  return success();
}

void* OclBufferImpl::GetNative(ErrorCode* ec) { return memory_->data(); }

size_t OclBufferImpl::GetSize(ErrorCode* ec) { return size_; }

////////////////////////////////////////////////////////////////////////////////
/// OclStreamImpl

OclStreamImpl::OclStreamImpl(Device device) : StreamImpl(device), queue_(), owned_queue_(false) {}

OclStreamImpl::~OclStreamImpl() {
  if (owned_queue_) {
    detail::Cast(queue_).~CommandQueue();
    queue_ = {};
    owned_queue_ = false;
  }
  external_.reset();
}

Result<void> OclStreamImpl::Init(uint64_t flags) {
  auto& platform = gOclPlatform();
  auto& ctx = platform.GetContext();
  auto& dev = platform.GetNativeDevice(device_.device_id());
  new (&queue_) cl::CommandQueue(ctx, dev);
  owned_queue_ = true;
  return success();
}

Result<void> OclStreamImpl::Init(std::shared_ptr<void> native, uint64_t flags) {
  external_ = std::move(native);
  queue_ = static_cast<cl_command_queue>(external_.get());
  owned_queue_ = false;
  return success();
}

Result<void> OclStreamImpl::DependsOn(Event& event) { return Status(eNotSupported); }

Result<void> OclStreamImpl::Query() {
  cl::Event event;
  queue().enqueueMarkerWithWaitList(nullptr, &event);
  auto status = event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
  if (status == CL_COMPLETE) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> OclStreamImpl::Wait() {
  auto status = queue().finish();
  if (status == CL_SUCCESS) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> OclStreamImpl::Submit(Kernel& kernel) { return Status(eNotSupported); }

void* OclStreamImpl::GetNative(ErrorCode* ec) { return queue_; }

////////////////////////////////////////////////////////////////////////////////
/// OclEventImpl

OclEventImpl::OclEventImpl(Device device) : EventImpl(device), event_(), owned_event_() {}

OclEventImpl::~OclEventImpl() = default;

Result<void> OclEventImpl::Init(uint64_t flags) { return success(); }

Result<void> OclEventImpl::Init(std::shared_ptr<void> native, uint64_t flags) {
  external_ = std::move(native);
  event_ = static_cast<cl_event>(external_.get());
  owned_event_ = false;
  return success();
}

Result<void> OclEventImpl::Query() {
  auto status = event().getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>();
  if (status == CL_COMPLETE) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> OclEventImpl::Record(Stream& stream) {
  auto& queue = Access::get<OclStreamImpl>(stream).queue();
  auto status = queue.enqueueMarkerWithWaitList(nullptr, &event());
  if (status == CL_SUCCESS) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> OclEventImpl::Wait() {
  if (!event_) {
    return success();
  }
  auto status = event().wait();
  if (status == CL_SUCCESS) {
    return success();
  } else {
    return Status(eFail);
  }
}

void* OclEventImpl::GetNative(ErrorCode* ec) { return event_; }

////////////////////////////////////////////////////////////////////////////////
/// OclPlatformRegisterer

class OclPlatformRegisterer {
 public:
  OclPlatformRegisterer() {
    gPlatformRegistry().Register([] {
      Logger::GetInstance().SetLogLevel(spdlog::level::debug);
      return std::make_shared<OclPlatformImpl>(cl::Platform::getDefault());
    });
  }
};

OclPlatformRegisterer g_ocl_platform_registerer;

}  // namespace mmdeploy

#endif
