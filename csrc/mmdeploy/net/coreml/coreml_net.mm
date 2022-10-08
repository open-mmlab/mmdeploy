// Copyright (c) OpenMMLab. All rights reserved.

#include "coreml_net.h"
#include "mmdeploy/core/model.h"
#include "mmdeploy/core/status_code.h"
#include "mmdeploy/core/utils/filesystem.h"
#include <fstream>

#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include <memory>

@interface MMBatchTensorFeatureProvider : NSObject <MLBatchProvider> {
  const std::vector<mmdeploy::Tensor> *inputs_;
}

- (instancetype)initWithInputs:(const std::vector<mmdeploy::Tensor> &)inputs;
- (NSInteger)count;
- (id<MLFeatureProvider>)featuresAtIndex:(NSInteger)index;
@end

@implementation MMBatchTensorFeatureProvider

- (instancetype)initWithInputs:(const std::vector<mmdeploy::Tensor> &)inputs {
  inputs_ = &inputs;
  return self;
}

- (NSInteger)count {
  return (*inputs_)[0].shape(0);
}

- (id<MLFeatureProvider>)featuresAtIndex:(NSInteger)index {
  MLDictionaryFeatureProvider *feature = nil;
  NSMutableDictionary<NSString *, id> *input_dict =
      [[NSMutableDictionary<NSString *, id> alloc] init];

  for (auto x : *inputs_) {
    auto in = x.Slice(index);
    NSMutableArray *shape = [[NSMutableArray alloc] init];
    for (const auto dim : in.shape()) {
      [shape addObject:[NSNumber numberWithLongLong:dim]];
    }

    NSMutableArray *strides = [[NSMutableArray alloc] init];
    int64_t stride = 1;
    for (int i = in.shape().size() - 1; i >= 0; i--) {
      [strides insertObject:[NSNumber numberWithLongLong:stride] atIndex:0];
      stride *= in.shape()[i];
    }

    MLMultiArrayDataType data_type = MLMultiArrayDataTypeFloat32;
    NSError *error = nil;
    MLMultiArray *mlArray =
        [[MLMultiArray alloc] initWithDataPointer:in.data()
                                            shape:shape
                                         dataType:data_type
                                          strides:strides
                                      deallocator:(^(void *){
                                                  })error:&error];
    if (error != nil) {
      MMDEPLOY_ERROR("init MLMultiArray failed with key: {}, error message: {}",
                     in.name(), [[error localizedDescription] UTF8String]);
      return nil;
    }

    NSString *key = [NSString stringWithUTF8String:in.name()];
    input_dict[key] = mlArray;
  }

  NSError *error = nil;
  feature = [[MLDictionaryFeatureProvider alloc] initWithDictionary:input_dict
                                                              error:&error];
  if (error != nil) {
    MMDEPLOY_ERROR("init MLDictionaryFeatureProvider failed with index: {}, "
                   "error message: {}",
                   index, [[error localizedDescription] UTF8String]);
    return nil;
  }

  return feature;
}

@end

namespace mmdeploy {
namespace coreml {

static Result<void> CheckInputOutputFeatureType(MLFeatureType type) {
  if (type != MLFeatureTypeMultiArray) {
    MMDEPLOY_ERROR("unsupported feature type: {}", type);
    return Status(eInvalidArgument);
  }
  return success();
}

static TensorShape to_shape(NSArray<NSNumber *> *shape) {
  TensorShape _shape;
  for (int i = 0; i < shape.count; i++) {
    _shape.push_back(shape[i].intValue);
  }
  return _shape;
}

static Result<DataType> ConvertElementType(MLMultiArrayDataType type) {
  switch (type) {
  case MLMultiArrayDataTypeFloat32:
    return DataType::kFLOAT;
  case MLMultiArrayDataTypeFloat16:
    return DataType::kHALF;
  case MLMultiArrayDataTypeInt32:
    return DataType::kINT32;
  default:
    MMDEPLOY_ERROR("unsupported MLMultiArrayDataType: {}",
                   static_cast<int>(type));
    return Status(eNotSupported);
  }
}

static Result<Tensor> AsTensor(MLMultiArray *mlArray, const Device &device) {
  TensorDesc desc;
  desc.device = device;
  desc.shape = to_shape(mlArray.shape);
  OUTCOME_TRY(desc.data_type, ConvertElementType(mlArray.dataType));
  std::shared_ptr<void> data(const_cast<void *>(mlArray.dataPointer),
                             [](void *) {});
  return Tensor(desc, data);
}

class Execution {
public:
  Execution(const std::string &path, CoreMLNet *net) : path_(path), net_(net) {}
  ~Execution() { RemoveModel(); }

  Result<void> Init() {
    OUTCOME_TRY(LoadModel());
    OUTCOME_TRY(SetInputOutputTensor());
    return success();
  }

  Result<void> Forward() {
    int batch_size = net_->input_tensors_[0].shape(0);

    // prepare input
    NSError *error = nil;
    MMBatchTensorFeatureProvider *input_feature =
        [[MMBatchTensorFeatureProvider alloc]
            initWithInputs:net_->input_tensors_];

    id<MLBatchProvider> output_feature =
        [model_ predictionsFromBatch:input_feature error:&error];
    if (error != nil) {
      MMDEPLOY_ERROR("coreml forward failed, error message: {}",
                     [[error localizedDescription] UTF8String]);
      return Status(eFail);
    }

    // extract output
    for (size_t i = 0; i < net_->output_tensors_.size(); ++i) {
      auto &out = net_->output_tensors_[i];

      for (int bid = 0; bid < output_feature.count; bid++) {
        NSString *name =
            [NSString stringWithCString:out.name()
                               encoding:[NSString defaultCStringEncoding]];
        if (name == nil) {
          MMDEPLOY_ERROR("output name must not be nil");
          return Status(eFail);
        }
        MLFeatureValue *output_value =
            [[output_feature featuresAtIndex:bid] featureValueForName:name];
        if (output_value == nil) {
          MMDEPLOY_ERROR("model output doesn't have name tensort: {}",
                         out.name());
          return Status(eFail);
        }

        MLMultiArray *mlArray = [output_value multiArrayValue];
        OUTCOME_TRY(auto tmp, AsTensor(mlArray, out.device()));
        if (bid == 0) {
          TensorShape batch_shape = tmp.shape();
          batch_shape[0] = batch_size;
          out.Reshape(batch_shape);
        }

        auto slice = out.Slice(bid);
        OUTCOME_TRY(tmp.CopyTo(slice, net_->stream_));
      }
    }

    return success();
  }

  Result<void> SetInputOutputTensor() {
    // input
    auto input_desc = model_.modelDescription.inputDescriptionsByName;
    for (NSString *name in input_desc) {
      MLFeatureDescription *value = input_desc[name];
      OUTCOME_TRY(CheckInputOutputFeatureType(value.type));
      // use default shape
      auto shape = to_shape(value.multiArrayConstraint.shape);
      OUTCOME_TRY(auto data_type,
                  ConvertElementType(value.multiArrayConstraint.dataType));
      net_->input_tensors_.emplace_back(
          TensorDesc{net_->device_, data_type, shape, [name UTF8String]});
    }

    // output
    auto output_desc = model_.modelDescription.outputDescriptionsByName;
    for (NSString *name in output_desc) {
      MLFeatureDescription *value = output_desc[name];
      OUTCOME_TRY(auto data_type,
                  ConvertElementType(value.multiArrayConstraint.dataType));
      // can't get output shape
      net_->output_tensors_.emplace_back(
          TensorDesc{net_->device_, data_type, {}, [name UTF8String]});
    }

    return success();
  }

  Result<void> Reshape(Span<TensorShape> input_shapes) {
    for (size_t i = 0; i < input_shapes.size(); ++i) {
      net_->input_tensors_[i].Reshape(input_shapes[i]);
    }
    return success();
  }

  Result<void> LoadModel() {
    NSString *model_path = [NSString stringWithUTF8String:path_.c_str()];
    NSError *error = nil;
    NSURL *model_url = [NSURL URLWithString:model_path];
    compiled_model_url_ = [MLModel compileModelAtURL:model_url error:&error];
    if (error != nil) {
      MMDEPLOY_ERROR("failed to compile model, error message: {}",
                     [[error localizedDescription] UTF8String]);
      return Status(eFail);
    }

    MLModelConfiguration *config = [MLModelConfiguration alloc];
    config.computeUnits = MLComputeUnitsAll;
    model_ = [MLModel modelWithContentsOfURL:compiled_model_url_
                               configuration:config
                                       error:&error];
    if (error != nil) {
      MMDEPLOY_ERROR("failed to construct model, error message: {}",
                     [[error localizedDescription] UTF8String]);
      return Status(eFail);
    }
    return success();
  }

  void RemoveModel() {
    NSError *error = nil;
    if (compiled_model_url_ != nil) {
      [[NSFileManager defaultManager] removeItemAtURL:compiled_model_url_
                                                error:&error];
      if (error != nil) {
        MMDEPLOY_ERROR("failed to remove compiled model, error message: {}",
                       [[error localizedDescription] UTF8String]);
      }
      compiled_model_url_ = nil;
    }
  }

  NSURL *compiled_model_url_{nil};
  MLModel *model_{nil};

  std::string path_;
  CoreMLNet *net_{nullptr};
};

} // namespace coreml

Result<void> CoreMLNet::Init(const Value &cfg) {
  auto &context = cfg["context"];
  device_ = context["device"].get<Device>();
  stream_ = context["stream"].get<Stream>();

  auto name = cfg["name"].get<std::string>();
  auto model = context["model"].get<Model>();
  OUTCOME_TRY(auto config, model.GetModelConfig(name));

  std::string coreml_tmp_path =
      (fs::path(model.GetModelPath()) / config.net).string();
  execution_ = std::make_unique<coreml::Execution>(coreml_tmp_path, this);
  OUTCOME_TRY(execution_->Init());

  return success();
}

Result<void> CoreMLNet::Deinit() { return success(); }

Result<Span<Tensor>> CoreMLNet::GetInputTensors() { return input_tensors_; }

Result<Span<Tensor>> CoreMLNet::GetOutputTensors() { return output_tensors_; }

Result<void> CoreMLNet::Reshape(Span<TensorShape> input_shapes) {
  return execution_->Reshape(input_shapes);
}

Result<void> CoreMLNet::Forward() { return execution_->Forward(); }

Result<void> CoreMLNet::ForwardAsync(Event *event) {
  return Status(eNotSupported);
}

class CoreMLNetCreator : public Creator<Net> {
public:
  const char *GetName() const override { return "coreml"; }
  int GetVersion() const override { return 0; }
  std::unique_ptr<Net> Create(const Value &args) override {
    auto p = std::make_unique<CoreMLNet>();
    if (auto r = p->Init(args)) {
      return p;
    } else {
      MMDEPLOY_ERROR("error creating CoreMLNet: {}",
                     r.error().message().c_str());
      return nullptr;
    }
  }
};

REGISTER_MODULE(Net, CoreMLNetCreator);

} // namespace mmdeploy
