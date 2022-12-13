// Copyright (c) OpenMMLab. All rights reserved.

#include <set>

#include "mmdeploy/archive/value_archive.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/operation/managed.h"
#include "mmdeploy/preprocess/elena/elena_registry.h"
#include "mmdeploy/preprocess/transform/tracer.h"
#include "mmdeploy/preprocess/transform/transform.h"

namespace mmdeploy::transform {

using namespace trace;
using namespace elena;

struct ExtractTransParamVisitor {
  bool valid{true};
  std::set<std::string> st;

  std::array<float, 3> mean;
  std::array<float, 3> std;
  std::array<int, 2> resize_hw;
  std::string resize_mode;
  float pad_val;
  std::array<int, 4> pad_tlbr;
  std::array<int, 2> pad_hw;
  std::array<int, 4> crop_tlbr;
  std::array<int, 2> crop_hw;

  void CheckValid(const std::string& name) {
    if (st.count(name)) {
      valid = false;
      return;
    }
    st.insert(name);
  }

  void operator()(CvtColorParam&) {}
  void operator()(CastParam&) {}
  void operator()(HWC2CHWParam&) {}

  void operator()(ResizeParam& param) {
    CheckValid("Resize");
    resize_hw = {param.size[0], param.size[1]};
    resize_mode = param.mode;
  }
  void operator()(PadParam& param) {
    CheckValid("Pad");
    pad_val = param.pad_val;
    std::copy_n(param.tlbr.begin(), 4, pad_tlbr.begin());
    std::copy_n(param.size.begin(), 2, pad_hw.begin());
  }
  void operator()(NormParam& param) {
    CheckValid("Normalize");
    std::copy(param.mean.begin(), param.mean.end(), mean.begin());
    std::copy(param.std.begin(), param.std.end(), std.begin());
  }
  void operator()(CropParam& param) {
    CheckValid("CenterCrop");
    std::copy_n(param.tlbr.begin(), 4, crop_tlbr.begin());
    std::copy_n(param.size.begin(), 2, crop_hw.begin());
  }
};

class Fused : public Transform {
 public:
  explicit Fused(const Value& args) {
    device_ = operation::gContext().device();
    tag_ = args["hash_code"].get<std::string>();
    tag_.append("_").append(GetPlatformName(device_));
    func_ = FuseKernel::Get().GetFunc(tag_);
    if (!func_) {
      MMDEPLOY_ERROR("can't find fuse function with tag: {}", tag_);
      throw_exception(eNotSupported);
    }
  }

  struct Context {
    Context() { operation::gContext().set_use_dummy(false); }
    ~Context() { operation::gContext().set_use_dummy(true); }
  };

  Result<void> Apply(Value& data) override {
    auto tracer = data["__tracer__"].get<Tracer>();
    Mat _src_mat = data["ori_img"].get<Mat>();

    auto& stream = operation::gContext().stream();

    // ! Create a scope that `use_dummy` is false
    Context context;
    OUTCOME_TRY(auto src_mat, operation::Secure(_src_mat, device_, stream));

    ExtractTransParamVisitor visitor{};
    for (auto&& trans : tracer.trans_) {
      std::visit(visitor, trans);
    }

    if (!visitor.valid) {
      MMDEPLOY_ERROR("unsupported fuse transform");
      return Status(eNotSupported);
    }
    if (src_mat.type() != DataType::kINT8) {
      MMDEPLOY_ERROR("unsupported data type in fuse transform");
      return Status(eNotSupported);
    }

    auto img_fields = GetImageFields(data);
    for (auto& key : img_fields) {
      assert(data.contains(key));
      auto src_tensor = data[key].get<Tensor>();
      auto desc = src_tensor.desc();
      desc.device = device_;
      Tensor dst_tensor{desc};

      func_(stream.GetNative(), src_mat.data<uint8_t>(), src_mat.height(), src_mat.width(),
            to_string(src_mat.pixel_format()).c_str(), visitor.resize_hw[0], visitor.resize_hw[1],
            visitor.resize_mode.c_str(), visitor.crop_tlbr[0], visitor.crop_tlbr[1],
            visitor.crop_hw[0], visitor.crop_hw[1], visitor.mean[0], visitor.mean[1],
            visitor.mean[2], visitor.std[0], visitor.std[1], visitor.std[2], visitor.pad_tlbr[0],
            visitor.pad_tlbr[1], visitor.pad_tlbr[2], visitor.pad_tlbr[3], visitor.pad_hw[0],
            visitor.pad_hw[1], visitor.pad_val, dst_tensor.data<float>(), dst_tensor.shape(2),
            dst_tensor.shape(3));
      operation::gContext().Track(dst_tensor);
      data[key] = std::move(dst_tensor);
    }
    return success();
  }

 private:
  Device device_;
  std::string tag_;
  FuseFunc func_;
};

MMDEPLOY_REGISTER_TRANSFORM(Fused);

}  // namespace mmdeploy::transform
