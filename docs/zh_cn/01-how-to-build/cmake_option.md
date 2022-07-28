# cmake 编译选项说明

<table class="docutils">
<thead>
  <tr>
    <th>选项</th>
    <th>取值范围</th>
    <th>缺省值</th>
    <th>说明</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>MMDEPLOY_BUILD_SDK</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK 编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_SDK_PYTHON_API</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK python package的编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_BUILD_TEST</td>
    <td>{ON, OFF}</td>
    <td>OFF</td>
    <td>MMDeploy SDK 的单元测试程序编译开关</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_DEVICES</td>
    <td>{"cpu", "cuda"}</td>
    <td>cpu</td>
    <td>设置目标设备。当有多个设备时，设备名称之间使用分号隔开。 比如，-DMMDEPLOY_TARGET_DEVICES="cpu;cuda"</td>
  </tr>
  <tr>
    <td>MMDEPLOY_TARGET_BACKENDS</td>
    <td>{"trt", "ort", "pplnn", "ncnn", "openvino"}</td>
    <td>N/A</td>
    <td> <b>默认情况下，SDK不设置任何后端</b>, 因为它与应用场景高度相关。 当选择多个后端时， 中间使用分号隔开。比如，<pre><code>-DMMDEPLOY_TARGET_BACKENDS="trt;ort;pplnn;ncnn;openvino"</code></pre>
    构建时，几乎每个后端，都需设置一些路径变量，用来查找依赖包。<br>
    1. <b>trt</b>: 表示 TensorRT。需要设置 <code>TENSORRT_DIR</code> 和 <code>CUDNN_DIR</code>。
<pre><code>
-DTENSORRT_DIR=$env:TENSORRT_DIR
-DCUDNN_DIR=$env:CUDNN_DIR
</code></pre>
    2. <b>ort</b>: 表示 ONNXRuntime。需要设置 <code>ONNXRUNTIME_DIR</code>。
<pre><code>-DONNXRUNTIME_DIR=$env:ONNXRUNTIME_DIR</code></pre>
    3. <b>pplnn</b>: 表示 PPL.NN。需要设置 <code>pplnn_DIR</code>。<b>当前版本尚未验证</b> <br>
    4. <b>ncnn</b>：表示 ncnn。需要设置 <code>ncnn_DIR</code>。<b>当前版本尚未验证</b> <br>
    5. <b>openvino</b>: 表示 OpenVINO。需要设置 <code>InferenceEngine_DIR</code>。<b>当前版本尚未验证通过</b>
   </td>
  </tr>
  <tr>
    <td>MMDEPLOY_CODEBASES</td>
    <td>{"mmcls", "mmdet", "mmseg", "mmedit", "mmocr", "all"}</td>
    <td>all</td>
    <td>用来设置SDK后处理组件，加载 OpenMMLab 算法仓库的后处理功能。如果选择多个 codebase，中间使用分号隔开。比如，<code>-DMMDEPLOY_CODEBASES="mmcls;mmdet"</code>。也可以通过 <code>-DMMDEPLOY_CODEBASES=all</code> 方式，加载所有 codebase。</td>
  </tr>
  <tr>
    <td>MMDEPLOY_SHARED_LIBS</td>
    <td>{ON, OFF}</td>
    <td>ON</td>
    <td>动态库的编译开关。设置OFF时，编译静态库</td>
  </tr>
</tbody>
</table>
