欢迎来到 MMDeploy 的中文文档！
====================================

点击页面左下角切换中英文。

.. toctree::
   :maxdepth: 2
   :caption: 快速上手

   get_started.md

.. toctree::
   :maxdepth: 1
   :caption: 编译

   01-how-to-build/build_from_source.md
   01-how-to-build/build_from_docker.md
   01-how-to-build/build_from_script.md
   01-how-to-build/cmake_option.md

.. toctree::
   :maxdepth: 1
   :caption: 运行和测试

   02-how-to-run/convert_model.md
   02-how-to-run/write_config.md
   02-how-to-run/profile_model.md
   02-how-to-run/quantize_model.md
   02-how-to-run/useful_tools.md

.. toctree::
   :maxdepth: 1
   :caption: Benchmark

   03-benchmark/supported_models.md
   03-benchmark/benchmark.md
   03-benchmark/benchmark_edge.md
   03-benchmark/benchmark_tvm.md
   03-benchmark/quantization.md

.. toctree::
   :maxdepth: 1
   :caption: 支持的算法框架

   04-supported-codebases/mmcls.md
   04-supported-codebases/mmdet.md
   04-supported-codebases/mmdet3d.md
   04-supported-codebases/mmedit.md
   04-supported-codebases/mmocr.md
   04-supported-codebases/mmpose.md
   04-supported-codebases/mmrotate.md
   04-supported-codebases/mmseg.md
   04-supported-codebases/mmaction2.md

.. toctree::
   :maxdepth: 1
   :caption: 支持的推理后端

   05-supported-backends/ncnn.md
   05-supported-backends/onnxruntime.md
   05-supported-backends/openvino.md
   05-supported-backends/pplnn.md
   05-supported-backends/rknn.md
   05-supported-backends/snpe.md
   05-supported-backends/tensorrt.md
   05-supported-backends/torchscript.md
   05-supported-backends/coreml.md

.. toctree::
   :maxdepth: 1
   :caption: 自定义算子

   06-custom-ops/ncnn.md
   06-custom-ops/onnxruntime.md
   06-custom-ops/tensorrt.md

.. toctree::
   :maxdepth: 1
   :caption: 开发者指南

   07-developer-guide/architecture.md
   07-developer-guide/support_new_model.md
   07-developer-guide/support_new_backend.md
   07-developer-guide/add_backend_ops_unittest.md
   07-developer-guide/test_rewritten_models.md
   07-developer-guide/partition_model.md
   07-developer-guide/regression_test.md

.. toctree::
   :maxdepth: 1
   :caption: 实验特性

   experimental/onnx_optimizer.md

.. toctree::
   :maxdepth: 1
   :caption: 新人解说

   tutorial/01_introduction_to_model_deployment.md
   tutorial/02_challenges.md
   tutorial/03_pytorch2onnx.md
   tutorial/04_onnx_custom_op.md
   tutorial/05_onnx_model_editing.md
   tutorial/06_introduction_to_tensorrt.md
   tutorial/07_write_a_plugin.md

.. toctree::
   :maxdepth: 1
   :caption: 附录

   appendix/cross_build_snpe_service.md

.. toctree::
   :maxdepth: 1
   :caption: 常见问题

   faq.md

.. toctree::
   :caption: 语言切换

   switch_language.md

.. toctree::
   :maxdepth: 1
   :caption: API 文档

   api.rst

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
