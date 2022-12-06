// Copyright (c) OpenMMLab. All rights reserved.
#include <iostream>
#include <set>
#include <string>

#include "elena_registry.h"
#include "mmdeploy/archive/json_archive.h"
#include "mmdeploy/core/mat.h"
#include "mmdeploy/core/tensor.h"
#include "mmdeploy/core/utils/device_utils.h"
#include "mmdeploy/core/utils/formatter.h"
#include "mmdeploy/preprocess/transform/collect.h"
#include "mmdeploy/preprocess/transform/tracer.h"

namespace mmdeploy::elena {

using namespace trace;

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

class CollectImpl : public ::mmdeploy::CollectImpl {
 public:
  CollectImpl(const Value& args) : ::mmdeploy::CollectImpl(args) {
    Platform platform(device_.platform_id());
    device_name_ = platform.GetPlatformName();
    sha256_ = args["context"].value("sha256", std::string(""));
  }

  ~CollectImpl() = default;

  Result<Value> Process(const Value& input) override {
    auto tracer = input["__tracer__"].get<Tracer>();
    Mat _src_mat = input["ori_img"].get<Mat>();
    OUTCOME_TRY(auto src_mat, MakeAvailableOnDevice(_src_mat, device_, stream_));
    OUTCOME_TRY(stream_.Wait());

    ExtractTransParamVisitor visitor{};
    for (auto&& trans : tracer.trans_) {
      std::visit(visitor, trans);
    }
    std::string tag = sha256_ + "_" + device_name_;
    FuseFunc func = FuseKernel::Get().GetFunc(tag);

    if (!visitor.valid) {
      MMDEPLOY_ERROR("unsupported fuse transform");
      throw std::invalid_argument("");
    }
    if (src_mat.type() != DataType::kINT8) {
      MMDEPLOY_ERROR("unsupported data type in fuse transform");
      throw std::invalid_argument("");
    }
    if (!func) {
      MMDEPLOY_ERROR("can't find fuse function with tag: {}", tag);
      throw std::invalid_argument("");
    }

    Value output = input;
    auto img_fields = GetImageFields(input);
    for (auto& key : img_fields) {
      assert(input.contains(key));
      auto src_tensor = input[key].get<Tensor>();
      auto desc = src_tensor.desc();
      desc.device = device_;
      Tensor dst_tensor{desc};

      func(stream_.GetNative(), src_mat.data<uint8_t>(), src_mat.height(), src_mat.width(),
           to_string(src_mat.pixel_format()).c_str(), visitor.resize_hw[0], visitor.resize_hw[1],
           visitor.resize_mode.c_str(), visitor.crop_tlbr[0], visitor.crop_tlbr[1],
           visitor.crop_hw[0], visitor.crop_hw[1], visitor.mean[0], visitor.mean[1],
           visitor.mean[2], visitor.std[0], visitor.std[1], visitor.std[2], visitor.pad_tlbr[0],
           visitor.pad_tlbr[1], visitor.pad_tlbr[2], visitor.pad_tlbr[3], visitor.pad_hw[0],
           visitor.pad_hw[1], visitor.pad_val, dst_tensor.data<float>(), dst_tensor.shape(2),
           dst_tensor.shape(3));
      output[key] = std::move(dst_tensor);
    }
    return ::mmdeploy::CollectImpl::Process(output);
  }

  std::string sha256_;
  std::string device_name_;
};

MMDEPLOY_REGISTER_TRANSFORM_IMPL(::mmdeploy::CollectImpl, (elena, 0), CollectImpl);

}  // namespace mmdeploy::elena
