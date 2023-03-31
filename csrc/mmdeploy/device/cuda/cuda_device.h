// Copyright (c) OpenMMLab. All rights reserved.

#include <any>
#include <mutex>

#include "cuda.h"
#include "cuda_runtime.h"
#include "mmdeploy/core/device_impl.h"
#include "mmdeploy/core/types.h"

namespace mmdeploy::framework {

using CudaTask = std::function<void(cudaStream_t)>;

class CudaPlatformImpl : public PlatformImpl {
 public:
  CudaPlatformImpl();

  ~CudaPlatformImpl() override {
    // The CUDA driver may have already shutdown before the platform dtor is called.
    // As a workaround, simply leak per device resources and let the driver handle it
    // FIXME: maybe a pair of global mmdeploy_init/deinit function would be a
    //  better solution
    for (auto& data : per_device_data_storage_) {
      data.release();
    }
  }

  const char* GetPlatformName() const noexcept override { return "cuda"; }

  Result<void> BindDevice(Device device, Device* prev) override;

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

  Allocator GetDefaultAllocator(int32_t device_id);

  Device GetDevice(int device_id) { return Device(platform_id_, device_id); }

 private:
  static bool CheckCopyParam(size_t src_size, size_t dst_size, size_t src_offset, size_t dst_offset,
                             size_t copy_size);

  static bool CheckCopyDevice(const Device& src, const Device& dst, const Device& st);

  static Result<void> CopyImpl(Device device, const void* src, void* dst, size_t src_size,
                               size_t dst_size, size_t src_offset, size_t dst_offset, size_t size,
                               Stream st);

  class PerDeviceData {
   public:
    explicit PerDeviceData(int device_id) : device_id_(device_id) {}
    void init();
    Stream& default_stream() {
      init();
      return default_stream_;
    }
    Allocator& default_allocator() {
      init();
      return default_allocator_;
    }

   private:
    int device_id_;
    std::once_flag init_flag_;
    Stream default_stream_;
    Allocator default_allocator_;
  };

  std::vector<std::unique_ptr<PerDeviceData>> per_device_data_storage_;
  std::vector<PerDeviceData*> per_device_data_;
};

CudaPlatformImpl& gCudaPlatform();

class CudaDeviceMemory;

class CudaBufferImpl : public BufferImpl {
 public:
  explicit CudaBufferImpl(Device device);

  Result<void> Init(size_t size, Allocator allocator, size_t alignment, uint64_t flags) override;

  Result<void> Init(size_t size, std::shared_ptr<void> native, uint64_t flags) override;

  Result<BufferImplPtr> SubBuffer(size_t offset, size_t size, uint64_t flags) override;

  void* GetNative(ErrorCode* ec) override;

  Allocator GetAllocator() const override;

  size_t GetSize(ErrorCode* ec) override;

 private:
  std::shared_ptr<CudaDeviceMemory> memory_;
  size_t offset_{0};
  size_t size_{0};
};

class CudaStreamImpl : public StreamImpl {
 public:
  explicit CudaStreamImpl(Device device);

  ~CudaStreamImpl() override;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> DependsOn(Event& event) override;

  Result<void> Query() override;

  Result<void> Wait() override;

  Result<void> Submit(Kernel& kernel) override;

  void* GetNative(ErrorCode* ec) override;

 private:
  cudaStream_t stream_;
  bool owned_stream_;
  std::shared_ptr<void> external_;
};

class CudaEventImpl : public EventImpl {
 public:
  explicit CudaEventImpl(Device device);

  ~CudaEventImpl() override;

  Result<void> Init(uint64_t flags) override;

  Result<void> Init(std::shared_ptr<void> native, uint64_t flags) override;

  Result<void> Query() override;

  Result<void> Record(Stream& stream) override;

  Result<void> Wait() override;

  void* GetNative(ErrorCode* ec) override;

 private:
  cudaEvent_t event_;
  bool owned_event_;
  std::shared_ptr<void> external_;
};

class CudaKernelImpl : public KernelImpl {
 public:
  explicit CudaKernelImpl(Device device, CudaTask task);

  void* GetNative(ErrorCode* ec) override;

 private:
  CudaTask task_;
};

class CudaDeviceGuard {
 public:
  explicit CudaDeviceGuard(Device device) : CudaDeviceGuard(device.device_id()) {}
  explicit CudaDeviceGuard(int device_id) : device_id_(device_id), prev_device_id_(-1) {
    CUcontext ctx{};
    cuCtxGetCurrent(&ctx);
    if (ctx) {
      cudaGetDevice(&prev_device_id_);
    }
    if (prev_device_id_ != device_id_) {
      cudaSetDevice(device_id_);
    }
  }
  ~CudaDeviceGuard() {
    if (prev_device_id_ >= 0 && prev_device_id_ != device_id_) {
      cudaSetDevice(prev_device_id_);
    }
  }

 private:
  int device_id_;
  int prev_device_id_;
};

}  // namespace mmdeploy::framework
