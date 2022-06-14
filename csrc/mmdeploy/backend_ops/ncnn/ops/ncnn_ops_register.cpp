// Copyright (c) OpenMMLab. All rights reserved.
#include "ncnn_ops_register.h"

#include <iostream>

std::map<const char *, ncnn::layer_creator_func> &get_mmdeploy_layer_creator() {
  static std::map<const char *, ncnn::layer_creator_func> _layer_creator_map;
  return _layer_creator_map;
}

std::map<const char *, ncnn::layer_destroyer_func> &get_mmdeploy_layer_destroyer() {
  static std::map<const char *, ncnn::layer_destroyer_func> _layer_destroyer_map;
  return _layer_destroyer_map;
}

int register_mmdeploy_custom_layers(ncnn::Net &net) {
  auto &layer_creator_map = get_mmdeploy_layer_creator();
  auto &layer_destroyer_map = get_mmdeploy_layer_destroyer();

  for (auto const &creator_pair : layer_creator_map) {
    auto creator_name = creator_pair.first;
    auto creator_func = creator_pair.second;

    ncnn::layer_destroyer_func destroyer_func = 0;
    if (layer_destroyer_map.find(creator_name) != layer_destroyer_map.end()) {
      destroyer_func = layer_destroyer_map[creator_name];
    }
    int ret = net.register_custom_layer(creator_name, creator_func, destroyer_func);
    if (0 != ret) {
      return ret;
    }
  }
  return 0;
}
