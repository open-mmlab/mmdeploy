// Copyright (c) OpenMMLab. All rights reserved.
#ifndef NCNN_OPS_DEFINER_H
#define NCNN_OPS_DEFINER_H

#include <string>

#include "layer.h"
#include "ncnn_ops_register.h"

namespace mmdeploy {

class NCNNOpsDefiner {
 public:
  NCNNOpsDefiner(const std::string& ops_name, const ncnn::layer_creator_func& creator_func = 0,
                 const ncnn::layer_destroyer_func& destroyer_func = 0)
      : _ops_name(ops_name) {
    get_mmdeploy_layer_creator()[_ops_name.c_str()] = creator_func;
  }

 private:
  const std::string _ops_name;
};

#define DEFINE_NCNN_OPS(ops_name, OpsLayer) \
  static mmdeploy::NCNNOpsDefiner NCNNOpsDefiner##ops_name{#ops_name, OpsLayer##_layer_creator};

}  // namespace mmdeploy

#endif
