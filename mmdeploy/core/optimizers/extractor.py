# Copyright (c) OpenMMLab. All rights reserved.
import re

import onnx
from packaging import version


def parse_extractor_io_string(io_str) -> tuple:
    """Parse IO string for extractor."""
    name, io_type = io_str.split(':')
    assert io_type in ['input', 'output']
    func_id = 0

    search_result = re.search(r'^(.+)\[([0-9]+)\]$', name)
    if search_result is not None:
        name = search_result.group(1)
        func_id = int(search_result.group(2))

    return name, func_id, io_type


def _dfs_search_reachable_nodes_fast(self, node_output_name, graph_input_nodes,
                                     reachable_nodes):
    """Using DFS to search reachable nodes."""
    outputs = {}
    for index, node in enumerate(self.graph.node):
        for name in node.output:
            if name not in outputs:
                outputs[name] = set()
            outputs[name].add(index)

    def impl(node_output_name, graph_input_nodes, reachable_nodes):
        if node_output_name in graph_input_nodes:
            return
        if node_output_name not in outputs:
            return
        for index in outputs[node_output_name]:
            node = self.graph.node[index]
            if node in reachable_nodes:
                continue
            reachable_nodes.append(node)
            for name in node.input:
                impl(name, graph_input_nodes, reachable_nodes)

    impl(node_output_name, graph_input_nodes, reachable_nodes)


def create_extractor(model: onnx.ModelProto) -> onnx.utils.Extractor:
    """Create Extractor for ONNX.

    Args:
        model (onnx.ModelProto): An input onnx model.

    Returns:
        onnx.utils.Extractor: Extractor for the onnx.
    """
    assert version.parse(onnx.__version__) >= version.parse('1.8.0')
    # patch extractor
    onnx.utils.Extractor._dfs_search_reachable_nodes = \
        _dfs_search_reachable_nodes_fast

    extractor = onnx.utils.Extractor(model)
    return extractor
