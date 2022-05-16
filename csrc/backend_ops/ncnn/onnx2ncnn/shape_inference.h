// Copyright (c) OpenMMLab. All rights reserved.

#pragma once

#include "utils.h"

/**
 * @brief query output shape of target node
 *
 * @param mutable_graph
 * @param target
 * @param weights
 * @param context <tensor name, shape>
 * @return std::tuple<bool, std::vector<int>>
 */
std::tuple<bool, std::vector<int>> query_shape(
    onnx::GraphProto* mutable_graph, onnx::NodeProto* target,
    const std::map<std::string, onnx::TensorProto>& weights,
    std::map<std::string, std::vector<int>>& context);
