// Copyright (c) OpenMMLab. All rights reserved.
#ifndef NCNN_OPS_REGISTER_H
#define NCNN_OPS_REGISTER_H

#include <map>
#include <string>

#include "mmdeploy/core/macro.h"
#include "net.h"

MMDEPLOY_API std::map<const char*, ncnn::layer_creator_func>& get_mmdeploy_layer_creator();
MMDEPLOY_API std::map<const char*, ncnn::layer_destroyer_func>& get_mmdeploy_layer_destroyer();

MMDEPLOY_API int register_mmdeploy_custom_layers(ncnn::Net& net);

#endif
