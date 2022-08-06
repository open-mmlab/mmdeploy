// Copyright (c) OpenMMLab. All rights reserved.

#include "cuda_device.h"

#include <cuda.h>

#include "mmdeploy/device/device_allocator.h"

namespace mmdeploy {

inline void* OffsetPtr(void* ptr, size_t offset) {
  return static_cast<void*>(static_cast<uint8_t*>(ptr) + offset);
}

inline const void* OffsetPtr(const void* ptr, size_t offset) {
  return static_cast<const void*>(static_cast<const uint8_t*>(ptr) + offset);
}

cudaMemcpyKind MapMemcpyKindToCuda(MemcpyKind kind) {
  switch (kind) {
    case MemcpyKind::HtoD:
      return cudaMemcpyHostToDevice;
    case MemcpyKind::DtoH:
      return cudaMemcpyDeviceToHost;
    case MemcpyKind::DtoD:
      return cudaMemcpyDeviceToDevice;
    default:
      return cudaMemcpyDefault;
  }
}

namespace cuda {

class Mallocator : public AllocatorImpl {
 public:
  Block Allocate(size_t size) noexcept override {
    if (size == 0) {
      return Block{};
    }
    Block block;
    if (auto status = cudaMalloc(&block.handle, size); status != cudaSuccess) {
      // log error
    }
    block.size = size;
    return block;
  }
  void Deallocate(Block& block) noexcept override {
    if (!block.handle) {
      return;
    }
    cudaFree(block.handle);
  }
  bool Owns(const Block& block) const noexcept override { return true; }
};

Allocator CreateDefaultAllocator() {
  using namespace device_allocator;
  AllocatorImplPtr allocator = std::make_shared<Mallocator>();
  allocator = std::make_shared<Tree>(allocator, -1, .5);
  allocator = std::make_shared<Locked>(allocator);
  MMDEPLOY_DEBUG("Default CUDA allocator initialized");
  return Access::create<Allocator>(allocator);
}

}  // namespace cuda

// ! this class doesn't handle device id
class CudaDeviceMemory : public NonCopyable {
 public:
  explicit CudaDeviceMemory(int device_id) : device_id_(device_id), size_(), owned_block_() {}
  Result<void> Init(size_t size, Allocator allocator, size_t alignment, uint64_t flags) {
    if (alignment != 1) {
      return Status(eNotSupported);
    }
    allocator_ = std::move(allocator);
    CudaDeviceGuard guard(device_id_);
    block_ = Access::get<AllocatorImpl>(allocator_).Allocate(size);
    if (size && !block_.handle) {
      return Status(eOutOfMemory);
    }
    size_ = size;
    owned_block_ = true;
    return success();
  }
  Result<void> Init(size_t size, std::shared_ptr<void> data, uint64_t flags) {
    size_ = size;
    external_ = std::move(data);
    block_.handle = external_.get();
    block_.size = size;
    owned_block_ = false;
    return success();
  }
  ~CudaDeviceMemory() {
    if (block_.handle) {
      if (owned_block_) {
        CudaDeviceGuard guard(device_id_);
        Access::get<AllocatorImpl>(allocator_).Deallocate(block_);
        owned_block_ = false;
      }
      block_.handle = nullptr;
    }
    external_.reset();
    size_ = 0;
  }
  size_t size() const { return size_; }
  void* data() const { return block_.handle; }
  const Allocator& allocator() const { return allocator_; }

 private:
  int device_id_;
  size_t size_;
  AllocatorImpl::Block block_;
  bool owned_block_;
  Allocator allocator_;
  std::shared_ptr<void> external_;
};

shared_ptr<BufferImpl> CudaPlatformImpl::CreateBuffer(Device device) {
  return std::make_shared<CudaBufferImpl>(device);
}

shared_ptr<StreamImpl> CudaPlatformImpl::CreateStream(Device device) {
  return std::make_shared<CudaStreamImpl>(device);
}

shared_ptr<EventImpl> CudaPlatformImpl::CreateEvent(Device device) {
  return std::make_shared<CudaEventImpl>(device);
}

bool CudaPlatformImpl::CheckCopyDevice(const Device& src, const Device& dst, const Device& st) {
  return st.is_device() && (src.is_host() || src == st) && (dst.is_host() || dst == st);
}

Result<void> CudaPlatformImpl::Copy(const void* host_ptr, Buffer dst, size_t size,
                                    size_t dst_offset, Stream stream) {
  if (!CheckCopyDevice(Device{0, 0}, dst.GetDevice(), stream.GetDevice())) {
    return Status(eInvalidArgument);
  }
  if (size == 0) {
    return success();
  }
  auto dst_ptr = dst.GetNative();
  if (!dst_ptr) {
    return Status(eInvalidArgument);
  }
  //  auto device = dst.GetDevice();
  return CopyImpl(stream.GetDevice(), host_ptr, dst_ptr, size, dst.GetSize(), 0, dst_offset, size,
                  stream);
}

Result<void> CudaPlatformImpl::Copy(Buffer src, void* host_ptr, size_t size, size_t src_offset,
                                    Stream stream) {
  if (!CheckCopyDevice(src.GetDevice(), Device{0, 0}, stream.GetDevice())) {
    return Status(eInvalidArgument);
  }
  if (size == 0) {
    return success();
  }
  auto src_ptr = src.GetNative();
  if (!src_ptr) {
    return Status(eInvalidArgument);
  }
  //  auto device = src.GetDevice();
  return CopyImpl(stream.GetDevice(), src_ptr, host_ptr, src.GetSize(), size, src_offset, 0, size,
                  stream);
}

Result<void> CudaPlatformImpl::Copy(Buffer src, Buffer dst, size_t size, size_t src_offset,
                                    size_t dst_offset, Stream stream) {
  if (!CheckCopyDevice(src.GetDevice(), dst.GetDevice(), stream.GetDevice())) {
    return Status(eInvalidArgument);
  }
  if (size == 0) {
    return success();
  }
  auto src_ptr = src.GetNative();
  auto dst_ptr = dst.GetNative();
  if (!src_ptr || !dst_ptr) {
    return Status(eInvalidArgument);
  }
  return CopyImpl(stream.GetDevice(), src_ptr, dst_ptr, src.GetSize(), dst.GetSize(), src_offset,
                  dst_offset, size, stream);
}

bool CudaPlatformImpl::CheckCopyParam(size_t src_size, size_t dst_size, size_t src_offset,
                                      size_t dst_offset, size_t copy_size) {
  if (src_offset + copy_size > src_size) {
    return false;
  }
  if (dst_offset + copy_size > dst_size) {
    return false;
  }
  return true;
}

Result<void> CudaPlatformImpl::CopyImpl(Device device, const void* src, void* dst, size_t src_size,
                                        size_t dst_size, size_t src_offset, size_t dst_offset,
                                        size_t size, Stream st) {
  if (!CheckCopyParam(src_size, dst_size, src_offset, dst_offset, size)) {
    return Status(eInvalidArgument);
  }

  auto p_dst = OffsetPtr(dst, dst_offset);
  auto p_src = OffsetPtr(src, src_offset);

  CudaDeviceGuard guard(device);

  if (st) {
    auto cuda_stream = ::mmdeploy::GetNative<cudaStream_t>(st);
    // TODO: how about default stream cudaStream_t(0)?
    if (!cuda_stream) {
      return Status(eInvalidArgument);
    }
    auto err = cudaMemcpyAsync(p_dst, p_src, size, cudaMemcpyDefault, cuda_stream);
    if (err != cudaSuccess) {
      return Status(eFail);
    }
  } else {
    auto err = cudaMemcpy(p_dst, p_src, size, cudaMemcpyDefault);
    if (err != cudaSuccess) {
      return Status(eFail);
    }
  }
  return success();
}

Result<Stream> CudaPlatformImpl::GetDefaultStream(int32_t device_id) {
  if (device_id >= per_device_data_.size()) {
    return Status(eInvalidArgument);
  }
  return per_device_data_[device_id]->default_stream();
}

void CudaPlatformImpl::PerDeviceData::init() {
  std::call_once(init_flag_, [&] {
    CudaDeviceGuard guard(device_id_);
    default_stream_ = Stream(gCudaPlatform().GetDevice(device_id_));
    default_allocator_ = cuda::CreateDefaultAllocator();
  });
}

CudaPlatformImpl::CudaPlatformImpl() {
  int count{};
  if (auto err = cudaGetDeviceCount(&count); err != cudaSuccess) {
    MMDEPLOY_ERROR("error getting device count: {}", cudaGetErrorString(err));
    throw_exception(eFail);
  }
  per_device_data_storage_.reserve(count);
  per_device_data_.reserve(count);
  for (int device_id = 0; device_id < count; ++device_id) {
    per_device_data_storage_.push_back(std::make_unique<PerDeviceData>(device_id));
    per_device_data_.push_back(per_device_data_storage_.back().get());
  }
}
Allocator CudaPlatformImpl::GetDefaultAllocator(int32_t device_id) {
  return per_device_data_[device_id]->default_allocator();
}

////////////////////////////////////////////////////////////////////////////////
/// CudaStreamImpl

CudaStreamImpl::CudaStreamImpl(Device device) : StreamImpl(device), stream_(), owned_stream_() {}

CudaStreamImpl::~CudaStreamImpl() {
  CudaDeviceGuard guard(device_.device_id());
  if (owned_stream_) {
    if (auto status = cudaStreamDestroy(stream_); status != cudaSuccess) {
      // TODO: signal error
    }
    owned_stream_ = false;
  }
  external_.reset();
}

Result<void> CudaStreamImpl::Init(uint64_t flags) {
  CudaDeviceGuard guard(device_);
  if (auto status = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
      status != cudaSuccess) {
    return Status(eFail);
  }
  owned_stream_ = true;
  return success();
}

Result<void> CudaStreamImpl::Init(std::shared_ptr<void> native, uint64_t flags) {
  // ! nullptr is valid for cudaStream_t
  external_ = std::move(native);
  stream_ = static_cast<cudaStream_t>(external_.get());
  owned_stream_ = false;
  return success();
}

Result<void> CudaStreamImpl::DependsOn(Event& event) {
  if (event.GetDevice() == device_) {
    CudaDeviceGuard guard(device_);
    auto native_event = ::mmdeploy::GetNative<cudaEvent_t>(event);
    cudaStreamWaitEvent(stream_, native_event, 0);
    return success();
  }
  return Status(eInvalidArgument);
}

Result<void> CudaStreamImpl::Query() {
  CudaDeviceGuard guard(device_);
  if (cudaStreamQuery(stream_) == cudaSuccess) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> CudaStreamImpl::Wait() {
  CudaDeviceGuard guard(device_);
  if (cudaStreamSynchronize(stream_) == cudaSuccess) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> CudaStreamImpl::Submit(Kernel& kernel) {
  auto task = ::mmdeploy::GetNative<CudaTask*>(kernel);
  if (task) {
    CudaDeviceGuard guard(device_);
    (*task)(stream_);
    return success();
  }
  return Status(eInvalidArgument);
}

void* CudaStreamImpl::GetNative(ErrorCode* ec) {
  if (ec) *ec = ErrorCode::eSuccess;
  return stream_;
}

////////////////////////////////////////////////////////////////////////////////
/// CudaEventImpl

CudaEventImpl::CudaEventImpl(Device device) : EventImpl(device), event_(), owned_event_() {}

CudaEventImpl::~CudaEventImpl() {
  CudaDeviceGuard guard(device_.device_id());
  if (owned_event_) {
    if (auto status = cudaEventDestroy(event_); status != cudaSuccess) {
      // TODO: signal error
    }
    owned_event_ = false;
  }
  external_.reset();
}

Result<void> CudaEventImpl::Init(uint64_t flags) {
  CudaDeviceGuard guard(device_);
  if (auto status = cudaEventCreateWithFlags(&event_, 0); status != cudaSuccess) {
    return Status(eFail);
  }
  owned_event_ = true;
  return success();
}

Result<void> CudaEventImpl::Init(std::shared_ptr<void> native, uint64_t flags) {
  if (!native) {
    return Status(eInvalidArgument);
  }
  external_ = std::move(native);
  event_ = static_cast<cudaEvent_t>(external_.get());
  owned_event_ = false;
  return success();
}

Result<void> CudaEventImpl::Query() {
  if (cudaEventQuery(event_) == cudaSuccess) {
    return success();
  } else {
    return Status(eFail);
  }
}

Result<void> CudaEventImpl::Record(Stream& stream) {
  if (stream.GetDevice() != device_) {
    return Status(eInvalidArgument);
  }
  CudaDeviceGuard guard(device_);
  auto native_stream = ::mmdeploy::GetNative<cudaStream_t>(stream);
  cudaEventRecord(event_, native_stream);
  return success();
}

Result<void> CudaEventImpl::Wait() {
  CudaDeviceGuard guard(device_);
  if (cudaEventSynchronize(event_) == cudaSuccess) {
    return success();
  } else {
    return Status(eFail);
  }
}

void* CudaEventImpl::GetNative(ErrorCode* ec) {
  if (ec) *ec = ErrorCode::eSuccess;
  return event_;
}
////////////////////////////////////////////////////////////////////////////////
/// CudaBufferImpl

CudaBufferImpl::CudaBufferImpl(Device device) : BufferImpl(device) {}

Result<void> CudaBufferImpl::Init(size_t size, Allocator allocator, size_t alignment,
                                  uint64_t flags) {
  memory_ = std::make_shared<CudaDeviceMemory>(device_.device_id());
  if (!allocator) {
    allocator = gCudaPlatform().GetDefaultAllocator(device_.device_id());
  }
  OUTCOME_TRY(memory_->Init(size, std::move(allocator), alignment, flags));
  size_ = size;
  return success();
}

Result<void> CudaBufferImpl::Init(size_t size, std::shared_ptr<void> native, uint64_t flags) {
  memory_ = std::make_shared<CudaDeviceMemory>(device_.device_id());
  OUTCOME_TRY(memory_->Init(size, std::move(native), flags));
  size_ = size;
  return success();
}

Result<BufferImplPtr> CudaBufferImpl::SubBuffer(size_t offset, size_t size, uint64_t flags) {
  if (offset_ + offset + size > memory_->size()) {
    return Status(eInvalidArgument);
  }
  auto impl = std::make_shared<CudaBufferImpl>(device_);
  impl->memory_ = memory_;
  impl->offset_ = offset_ + offset;
  impl->size_ = size;
  return impl;
}

size_t CudaBufferImpl::GetSize(ErrorCode* ec) { return size_; }

void* CudaBufferImpl::GetNative(ErrorCode* ec) {
  if (!memory_) {
    if (ec) *ec = eInvalidArgument;
    return nullptr;
  }
  if (ec) *ec = ErrorCode::eSuccess;
  return OffsetPtr(memory_->data(), offset_);
}

Allocator CudaBufferImpl::GetAllocator() const { return memory_->allocator(); }

////////////////////////////////////////////////////////////////////////////////
/// CudaKernelImpl
void* CudaKernelImpl::GetNative(ErrorCode* ec) {
  if (ec) *ec = ErrorCode::eSuccess;
  return &task_;
}

CudaKernelImpl::CudaKernelImpl(Device device, CudaTask task)
    : KernelImpl(device), task_(std::move(task)) {}

////////////////////////////////////////////////////////////////////////////////
/// CudaPlatformRegisterer
class CudaPlatformRegisterer {
 public:
  CudaPlatformRegisterer() {
    gPlatformRegistry().Register([] { return std::make_shared<CudaPlatformImpl>(); });
  }
};

CudaPlatformRegisterer g_cuda_platform_registerer;

CudaPlatformImpl& gCudaPlatform() {
  static Platform platform("cuda");
  return Access::get<CudaPlatformImpl>(platform);
}

}  // namespace mmdeploy
