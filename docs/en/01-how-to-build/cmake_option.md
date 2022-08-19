# CMake Build Option Spec

<table class="docutils">
<thead>
  <tr>
    <th>NAME</th>
    <th>VALUE</th>
    <th>DEFAULT</th>
    <th>REMARK</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MMDEPLOY_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>Switch to build shared libs</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_SDK</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Switch to build MMDeploy SDK</td>
  </tr>

<tr>
  <td>MMDEPLOY_BUILD_SDK_MONOLITHIC</td>
  <td>{ON, OFF}</td>
  <td>OFF</td>
  <td>Build single lib</td>
  </tr>

<tr>
  <td>MMDEPLOY_BUILD_TEST</td>
  <td>{ON, OFF}</td>
  <td>OFF</td>
  <td>Build unittest</td>
  </tr>

<tr>
    <td>MMDEPLOY_BUILD_SDK_PYTHON_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Switch to build MMDeploy SDK python package</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_SDK_CXX_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Build C++ SDK API</td>
  </tr>

<tr>
    <td>MMDEPLOY_BUILD_SDK_CSHARP_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Build C# SDK API</td>
  </tr>

<tr>
    <td>MMDEPLOY_BUILD_SDK_JAVA_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Build Java SDK API</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_TEST</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Switch to build MMDeploy SDK unittest cases</td>
  </tr>

<tr>
    <td>MMDEPLOY_SPDLOG_EXTERNAL</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Build with spdlog installation package that comes with the system</td>
  </tr>

<tr>
    <td>MMDEPLOY_ZIP_MODEL</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Enable SDK with zip format</td>
  </tr>

<tr>
    <td>MMDEPLOY_COVERAGE</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>Build for cplus code coverage report</td>
  </tr>

<tr>
    <td>MMDEPLOY_TARGET_DEVICES</td>
    <td>{"cpu", "cuda"}</td>
    <td>cpu</td>
    <td>Enable target device. You can enable more by
   passing a semicolon separated list of device names to <code>MMDEPLOY_TARGET_DEVICES</code> variable, e.g. <code>-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"</code> </td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_BACKENDS</td>
    <td>{"trt", "ort", "pplnn", "ncnn", "openvino", "torchscript", "snpe"}</td>
    <td>N/A</td>
    <td>Enabling inference engine. <b>By default, no target inference engine is set, since it highly depends on the use case.</b> When more than one engine are specified, it has to be set with a semicolon separated list of inference backend names, e.g. <pre><code>-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"</code></pre>
    After specifying the inference engine, it's package path has to be passed to cmake as follows, <br>
    1. <b>trt</b>: TensorRT. <code>TENSORRT_DIR</code> and <code>CUDNN_DIR</code> are needed.
<pre><code>
-DTENSORRT_DIR=${TENSORRT_DIR}
-DCUDNN_DIR=${CUDNN_DIR}
</code></pre>
    2. <b>ort</b>: ONNXRuntime. <code>ONNXRUNTIME_DIR</code> is needed.
<pre><code>-DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR}</code></pre>
    3. <b>pplnn</b>: PPL.NN. <code>pplnn_DIR</code> is needed.
<pre><code>-Dpplnn_DIR=${PPLNN_DIR}</code></pre>
    4. <b>ncnn</b>: ncnn. <code>ncnn_DIR</code> is needed.
<pre><code>-Dncnn_DIR=${NCNN_DIR}/build/install/lib/cmake/ncnn</code></pre>
    5. <b>openvino</b>: OpenVINO. <code>InferenceEngine_DIR</code> is needed.
<pre><code>-DInferenceEngine_DIR=${INTEL_OPENVINO_DIR}/deployment_tools/inference_engine/share</code></pre>
    6. <b>torchscript</b>: TorchScript. <code>Torch_DIR</code> is needed.
<pre><code>-DTorch_DIR=${Torch_DIR}</code></pre>
Currently, <b>The Model Converter supports torchscript, but SDK doesn't</b>.<br>
    7. <b>snpe</b>: qcom snpe. <code>SNPE_ROOT</code> must existed in the environment variable because of C/S mode.
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>all</td>
    <td>Enable codebase's postprocess modules. You can provide a semicolon separated list of codebase names to enable them, e.g., <code>-DMMDEPLOY_CODEBASES="mmcls;mmdet"</code>. Or you can pass <code>all</code> to enable them all, i.e., <code>-DMMDEPLOY_CODEBASES=all</code></td>
  </tr>

</tbody>
</table>
