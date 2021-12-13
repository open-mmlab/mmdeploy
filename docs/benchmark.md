## Benchmark

### Backends
CPU: ncnn, ONNXRuntime
GPU: TensorRT, PPLNN

### Platform
- Ubuntu 18.04
- Cuda 11.3
- TensorRT 7.2.3.4
- Docker 20.10.8
- NVIDIA tesla T4 tensor core GPU for TensorRT.

### Other settings
- Static graph
- Batch size 1
- Synchronize devices after each inference.
- We count the average inference performance of 100 images of the dataset.
- Warm up. For classification, we warm up 1010 iters. For other codebases, we warm up 10 iters.
- Input resolution varies for different datasets of different codebases. All inputs are real images except for mmediting because the dataset is not large enough.

### Latency benchmark
Users can directly test the speed through [how_to_measure_performance_of_models.md](docs/tutorials/how_to_measure_performance_of_models.md). And here is the benchmark in our environment.
<details>
<summary style="margin-left: 25px;">MMCls with 1x3x224x224 input</summary>
<div style="margin-left: 25px;">

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-c3ow" colspan="2">PPLNN</th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Model</td>
    <td class="tg-9wq8" rowspan="2">Input</td>
    <td class="tg-c3ow" colspan="2">fp32</td>
    <td class="tg-c3ow" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow" colspan="2">in8</td>
    <td class="tg-c3ow" colspan="2">fp16</td>
    <td class="tg-lboi" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNet</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">2.97</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">336.90</span></td>
    <td class="tg-c3ow">1.26</td>
    <td class="tg-c3ow">791.89</td>
    <td class="tg-c3ow">1.21</td>
    <td class="tg-c3ow">829.66</td>
    <td class="tg-c3ow">1.30</td>
    <td class="tg-c3ow">768.28</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/resnet/resnet50_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ResNeXt</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">4.31</td>
    <td class="tg-c3ow">231.93</td>
    <td class="tg-c3ow">1.42</td>
    <td class="tg-c3ow">703.42</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">727.42</td>
    <td class="tg-c3ow">1.36</td>
    <td class="tg-c3ow">737.67</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/resnext/resnext50_32x4d_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">SE-ResNet</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">3.41</td>
    <td class="tg-c3ow">293.64</td>
    <td class="tg-c3ow">1.66</td>
    <td class="tg-c3ow">600.73</td>
    <td class="tg-c3ow">1.51</td>
    <td class="tg-c3ow">662.90</td>
    <td class="tg-c3ow">1.91</td>
    <td class="tg-c3ow">524.07</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/seresnet/seresnet50_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">ShuffleNetV2</td>
    <td class="tg-c3ow">1x3x224x224</td>
    <td class="tg-c3ow">1.37</td>
    <td class="tg-c3ow">727.94</td>
    <td class="tg-c3ow">1.19</td>
    <td class="tg-c3ow">841.36</td>
    <td class="tg-c3ow">1.13</td>
    <td class="tg-c3ow">883.47</td>
    <td class="tg-c3ow">4.69</td>
    <td class="tg-c3ow">213.33</td>
    <td class="tg-0pky">$MMCLS_DIR/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMEditing with 1x3x32x32 input</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh" colspan="2"></th>
    <th class="tg-baqh" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-baqh" colspan="2">PPLNN</th>
    <th class="tg-0lax"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="2">Model</td>
    <td class="tg-nrix" rowspan="2">Input</td>
    <td class="tg-baqh" colspan="2">fp32</td>
    <td class="tg-baqh" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-baqh" colspan="2">in8</td>
    <td class="tg-baqh" colspan="2">fp16</td>
    <td class="tg-cly1" rowspan="2"><span style="font-weight:400;font-style:normal">model config file</span></td>
  </tr>
  <tr>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
  </tr>
  <tr>
    <td class="tg-baqh">ESRGAN</td>
    <td class="tg-baqh">1x3x32x32</td>
    <td class="tg-baqh">12.64</td>
    <td class="tg-baqh">79.14</td>
    <td class="tg-baqh">12.42</td>
    <td class="tg-baqh">80.50</td>
    <td class="tg-baqh">12.45</td>
    <td class="tg-baqh">80.35</td>
    <td class="tg-baqh">7.67</td>
    <td class="tg-baqh">130.39</td>
    <td class="tg-0lax">$MMEDIT_DIR/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">SRCNN</td>
    <td class="tg-baqh">1x3x32x32</td>
    <td class="tg-baqh">0.70</td>
    <td class="tg-baqh">1436.47</td>
    <td class="tg-baqh">0.35</td>
    <td class="tg-baqh">2836.62</td>
    <td class="tg-baqh">0.26</td>
    <td class="tg-baqh">3850.45</td>
    <td class="tg-baqh">0.56</td>
    <td class="tg-baqh">1775.11</td>
    <td class="tg-0lax">$MMEDIT_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMSeg with 1x3x512x1024 input</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2"></th>
    <th class="tg-c3ow" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-c3ow" colspan="2">PPLNN</th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Model</td>
    <td class="tg-9wq8" rowspan="2">Input</td>
    <td class="tg-c3ow" colspan="2">fp32</td>
    <td class="tg-c3ow" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow" colspan="2">in8</td>
    <td class="tg-c3ow" colspan="2">fp16</td>
    <td class="tg-lboi" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
    <td class="tg-c3ow">latency (ms)</td>
    <td class="tg-c3ow">FPS</td>
  </tr>
  <tr>
    <td class="tg-c3ow">FCN</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">128.42</td>
    <td class="tg-c3ow">7.79</td>
    <td class="tg-c3ow">23.97</td>
    <td class="tg-c3ow">41.72</td>
    <td class="tg-c3ow">18.13</td>
    <td class="tg-c3ow">55.15</td>
    <td class="tg-c3ow">27.00</td>
    <td class="tg-c3ow">37.04</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">PSPNet</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">119.77</td>
    <td class="tg-c3ow">8.35</td>
    <td class="tg-c3ow">24.10</td>
    <td class="tg-c3ow">41.49</td>
    <td class="tg-c3ow">16.33</td>
    <td class="tg-c3ow">61.23</td>
    <td class="tg-c3ow">27.26</td>
    <td class="tg-c3ow">36.69</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DeepLabV3</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">226.75</td>
    <td class="tg-c3ow">4.41</td>
    <td class="tg-c3ow">31.80</td>
    <td class="tg-c3ow">31.45</td>
    <td class="tg-c3ow">19.85</td>
    <td class="tg-c3ow">50.38</td>
    <td class="tg-c3ow">36.01</td>
    <td class="tg-c3ow">27.77</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-c3ow">DeepLabV3+</td>
    <td class="tg-c3ow">1x3x512x1024</td>
    <td class="tg-c3ow">151.25</td>
    <td class="tg-c3ow">6.61</td>
    <td class="tg-c3ow">47.03</td>
    <td class="tg-c3ow">21.26</td>
    <td class="tg-c3ow">50.38</td>
    <td class="tg-c3ow">26.67</td>
    <td class="tg-c3ow">34.80</td>
    <td class="tg-c3ow">28.74</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMDet with 1x3x800x1344 input</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh" colspan="2"></th>
    <th class="tg-baqh" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-baqh" colspan="2">PPLNN</th>
    <th class="tg-0lax"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="2">Model</td>
    <td class="tg-nrix" rowspan="2">Input</td>
    <td class="tg-baqh" colspan="2">fp32</td>
    <td class="tg-baqh" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-baqh" colspan="2">in8</td>
    <td class="tg-baqh" colspan="2">fp16</td>
    <td class="tg-cly1" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
  </tr>
  <tr>
    <td class="tg-baqh">YOLOv3</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">94.08</td>
    <td class="tg-baqh">10.63</td>
    <td class="tg-baqh">24.90</td>
    <td class="tg-baqh">40.17</td>
    <td class="tg-baqh">24.87</td>
    <td class="tg-baqh">40.21</td>
    <td class="tg-baqh">47.64</td>
    <td class="tg-baqh">20.99</td>
    <td class="tg-0lax">$MMDET_DIR/configs/yolo/yolov3_d53_320_273e_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">SSD-Lite</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">14.91</td>
    <td class="tg-baqh">67.06</td>
    <td class="tg-baqh">8.92</td>
    <td class="tg-baqh">112.13</td>
    <td class="tg-baqh">8.65</td>
    <td class="tg-baqh">115.63</td>
    <td class="tg-baqh">30.13</td>
    <td class="tg-baqh">33.19</td>
    <td class="tg-0lax">$MMDET_DIR/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">RetinaNet</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">97.09</td>
    <td class="tg-baqh">10.30</td>
    <td class="tg-baqh">25.79</td>
    <td class="tg-baqh">38.78</td>
    <td class="tg-baqh">16.88</td>
    <td class="tg-baqh">59.23</td>
    <td class="tg-baqh">38.34</td>
    <td class="tg-baqh">26.08</td>
    <td class="tg-0lax">$MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">FCOS</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">84.06</td>
    <td class="tg-baqh">11.90</td>
    <td class="tg-baqh">23.15</td>
    <td class="tg-baqh">43.20</td>
    <td class="tg-baqh">17.68</td>
    <td class="tg-baqh">56.57</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-0lax">$MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">FSAF</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">82.96</td>
    <td class="tg-baqh">12.05</td>
    <td class="tg-baqh">21.02</td>
    <td class="tg-baqh">47.58</td>
    <td class="tg-baqh">13.50</td>
    <td class="tg-baqh">74.08</td>
    <td class="tg-baqh">30.41</td>
    <td class="tg-baqh">32.89</td>
    <td class="tg-0lax">$MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">Faster-RCNN</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">88.08</td>
    <td class="tg-baqh">11.35</td>
    <td class="tg-baqh">26.52</td>
    <td class="tg-baqh">37.70</td>
    <td class="tg-baqh">19.14</td>
    <td class="tg-baqh">52.23</td>
    <td class="tg-baqh">65.40</td>
    <td class="tg-baqh">15.29</td>
    <td class="tg-0lax">$MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">Mask-RCNN</td>
    <td class="tg-baqh">1x3x800x1344</td>
    <td class="tg-baqh">320.86 </td>
    <td class="tg-baqh">3.12</td>
    <td class="tg-baqh">241.32</td>
    <td class="tg-baqh">4.14</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">86.80</td>
    <td class="tg-baqh">11.52</td>
    <td class="tg-0lax">$MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMOCR</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh" colspan="2"></th>
    <th class="tg-baqh" colspan="6"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-baqh" colspan="2">PPLNN</th>
    <th class="tg-0lax"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-nrix" rowspan="2">Model</td>
    <td class="tg-nrix" rowspan="2">Input</td>
    <td class="tg-baqh" colspan="2">fp32</td>
    <td class="tg-baqh" colspan="2"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-baqh" colspan="2">in8</td>
    <td class="tg-baqh" colspan="2">fp16</td>
    <td class="tg-cly1" rowspan="2">model config file</td>
  </tr>
  <tr>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
    <td class="tg-baqh">latency (ms)</td>
    <td class="tg-baqh">FPS</td>
  </tr>
  <tr>
    <td class="tg-baqh">DBNet</td>
    <td class="tg-baqh">1x3x640x640</td>
    <td class="tg-baqh">10.70</td>
    <td class="tg-baqh">93.43</td>
    <td class="tg-baqh">5.62</td>
    <td class="tg-baqh">177.78</td>
    <td class="tg-baqh">5.00</td>
    <td class="tg-baqh">199.85</td>
    <td class="tg-baqh">34.84</td>
    <td class="tg-baqh">28.70</td>
    <td class="tg-0lax">$MMOCR_DIR/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">CRNN</td>
    <td class="tg-baqh">1x1x32x32</td>
    <td class="tg-baqh">1.93 </td>
    <td class="tg-baqh">518.28</td>
    <td class="tg-baqh">1.40</td>
    <td class="tg-baqh">713.88</td>
    <td class="tg-baqh">1.36</td>
    <td class="tg-baqh">736.79</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-0lax">$MMOCR_DIR/configs/textrecog/crnn/crnn_academic_dataset.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

### Performance benchmark

Users can directly test the performance through [how_to_evaluate_a_model.md](docs/tutorials/how_to_evaluate_a_model.md). And here is the benchmark in our environment.

<details>
<summary style="margin-left: 25px;">MMEditing</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="3">MMEditing</th>
    <th class="tg-0lax">PyTorch</th>
    <th class="tg-0pky">ONNX Runtime</th>
    <th class="tg-c3ow" colspan="3"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-c3ow">PPLNN</th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8">Model</td>
    <td class="tg-9wq8">Task</td>
    <td class="tg-0pky">Metrics(Set5)</td>
    <td class="tg-baqh">fp32</td>
    <td class="tg-c3ow">fp32</td>
    <td class="tg-c3ow">fp32</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow">int8</td>
    <td class="tg-c3ow">fp16</td>
    <td class="tg-lboi">model config file</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">SRCNN</td>
    <td class="tg-9wq8" rowspan="2">Super Resolution</td>
    <td class="tg-0pky">PSNR</td>
    <td class="tg-0lax">28.4316</td>
    <td class="tg-c3ow">28.4323</td>
    <td class="tg-c3ow">28.4323</td>
    <td class="tg-c3ow">28.4286</td>
    <td class="tg-c3ow">28.1995</td>
    <td class="tg-c3ow">28.4311</td>
    <td class="tg-lboi" rowspan="2">$MMEDIT_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0lax">0.8099</td>
    <td class="tg-c3ow">0.8097</td>
    <td class="tg-c3ow">0.8097</td>
    <td class="tg-c3ow">0.8096</td>
    <td class="tg-c3ow">0.7934</td>
    <td class="tg-c3ow">0.8096</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">ESRGAN</td>
    <td class="tg-9wq8" rowspan="2">Super Resolution</td>
    <td class="tg-0pky">PSNR</td>
    <td class="tg-0lax">28.2700</td>
    <td class="tg-c3ow">28.2592</td>
    <td class="tg-c3ow">28.2592</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">28.2624</td>
    <td class="tg-lboi" rowspan="2">$MMEDIT_DIR/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0lax">0.7778</td>
    <td class="tg-c3ow">0.7764</td>
    <td class="tg-c3ow">0.7774</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">0.7765</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="2">ESRGAN-PSNR</td>
    <td class="tg-9wq8" rowspan="2">Super Resolution</td>
    <td class="tg-0pky">PSNR</td>
    <td class="tg-0lax">30.6428</td>
    <td class="tg-c3ow">30.6444</td>
    <td class="tg-c3ow">30.6430</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">27.0426</td>
    <td class="tg-lboi" rowspan="2">$MMEDIT_DIR/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0lax">0.8559</td>
    <td class="tg-c3ow">0.8558</td>
    <td class="tg-c3ow">0.8558</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">0.8557</td>
  </tr>
    <tr>
    <td class="tg-9wq8" rowspan="2">SRGAN</td>
    <td class="tg-9wq8" rowspan="2">Super Resolution</td>
    <td class="tg-0pky">PSNR</td>
    <td class="tg-0lax">27.9499</td>
    <td class="tg-c3ow">27.9408</td>
    <td class="tg-c3ow">27.9408</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">27.9388</td>
    <td class="tg-lboi" rowspan="2">$MMEDIT_DIR/configs/restorers/srresnet_srgan/srgan_x4c64b16_g1_1000k_div2k.pyy</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0lax">0.7846</td>
    <td class="tg-c3ow">0.7839</td>
    <td class="tg-c3ow">0.7839</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">0.7839</td>
  </tr>
    </tr>
    <tr>
    <td class="tg-9wq8" rowspan="2">SRResNet</td>
    <td class="tg-9wq8" rowspan="2">Super Resolution</td>
    <td class="tg-0pky">PSNR</td>
    <td class="tg-0lax">30.2252</td>
    <td class="tg-c3ow">30.2300</td>
    <td class="tg-c3ow">30.2300</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">30.2294</td>
    <td class="tg-lboi" rowspan="2">$MMEDIT_DIR/configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0lax">0.8491</td>
    <td class="tg-c3ow">0.8488</td>
    <td class="tg-c3ow">0.8488</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">0.8488</td>
  </tr>
    </tr>
    </tr>
    <tr>
    <td class="tg-9wq8" rowspan="2">Real-ESRNet</td>
    <td class="tg-9wq8" rowspan="2">Super Resolution</td>
    <td class="tg-0pky">PSNR</td>
    <td class="tg-0lax">28.0297</td>
    <td class="tg-c3ow">27.7016</td>
    <td class="tg-c3ow">27.7016</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">27.7049</td>
    <td class="tg-lboi" rowspan="2">$MMEDIT_DIR/configs/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0lax">0.8236</td>
    <td class="tg-c3ow">0.8122</td>
    <td class="tg-c3ow">0.8122</td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow"> - </td>
    <td class="tg-c3ow">0.8123</td>
  </tr>
  </tr>
    </tr>
    </tr>
    <tr>
    <td class="tg-9wq8" rowspan="2">EDSR</td>
    <td class="tg-9wq8" rowspan="2">Super Resolution</td>
    <td class="tg-0pky">PSNR</td>
    <td class="tg-0lax">30.2223</td>
    <td class="tg-c3ow">30.2214</td>
    <td class="tg-c3ow">30.2214</td>
    <td class="tg-c3ow">30.2211</td>
    <td class="tg-c3ow">30.1383</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-lboi" rowspan="2">$MMEDIT_DIR/configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">SSIM</td>
    <td class="tg-0lax">0.8500</td>
    <td class="tg-c3ow">0.8497</td>
    <td class="tg-c3ow">0.8497</td>
    <td class="tg-c3ow">0.8497</td>
    <td class="tg-c3ow">0.8469</td>
    <td class="tg-c3ow"> - </td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMOCR</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-baqh" colspan="3">MMOCR</th>
    <th class="tg-baqh">Pytorch</th>
    <th class="tg-baqh">ONNXRuntime</th>
    <th class="tg-baqh" colspan="3"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-baqh">PPLNN</th>
    <th class="tg-0lax">OpenVINO</th>
    <th class="tg-0lax"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-baqh">Model</td>
    <td class="tg-baqh">Task</td>
    <td class="tg-baqh">Metrics</td>
    <td class="tg-baqh">fp32</td>
    <td class="tg-baqh">fp32</td>
    <td class="tg-baqh">fp32</td>
    <td class="tg-baqh"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-baqh">int8</td>
    <td class="tg-baqh">fp16</td>
    <td class="tg-baqh">fp32</td>
    <td class="tg-0lax">model config file</td>
  </tr>
  <tr>
    <td class="tg-nrix" rowspan="3">DBNet*</td>
    <td class="tg-nrix" rowspan="3">TextDetection</td>
    <td class="tg-baqh">recall</td>
    <td class="tg-baqh">0.7310</td>
    <td class="tg-baqh">0.7304</td>
    <td class="tg-baqh">0.7198</td>
    <td class="tg-baqh">0.7179</td>
    <td class="tg-baqh">0.7111</td>
    <td class="tg-baqh">0.7304</td>
    <td class="tg-0lax">0.7309</td>
    <td class="tg-cly1" rowspan="3">$MMOCR_DIR/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">precision</td>
    <td class="tg-baqh">0.8714</td>
    <td class="tg-baqh">0.8718</td>
    <td class="tg-baqh">0.8677</td>
    <td class="tg-baqh">0.8674</td>
    <td class="tg-baqh">0.8688</td>
    <td class="tg-baqh">0.8718</td>
    <td class="tg-0lax">0.8714</td>
  </tr>
  <tr>
    <td class="tg-baqh">hmean</td>
    <td class="tg-baqh">0.7950</td>
    <td class="tg-baqh">0.7949</td>
    <td class="tg-baqh">0.7868</td>
    <td class="tg-baqh">0.7856</td>
    <td class="tg-baqh">0.7821</td>
    <td class="tg-baqh">0.7949</td>
    <td class="tg-0lax">0.7950</td>
  </tr>
  <tr>
    <td class="tg-baqh">CRNN</td>
    <td class="tg-baqh">TextRecognition</td>
    <td class="tg-baqh">acc</td>
    <td class="tg-baqh">0.8067</td>
    <td class="tg-baqh">0.8067</td>
    <td class="tg-baqh">0.8067</td>
    <td class="tg-baqh">0.8063</td>
    <td class="tg-baqh">0.8067</td>
    <td class="tg-baqh">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">$MMOCR_DIR/configs/textrecog/crnn/crnn_academic_dataset.py</td>
  </tr>
  <tr>
    <td class="tg-baqh">SAR</td>
    <td class="tg-baqh">TextRecognition</td>
    <td class="tg-baqh">acc</td>
    <td class="tg-baqh">0.9517</td>
    <td class="tg-baqh">0.9287</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-baqh">-</td>
    <td class="tg-0lax">-</td>
    <td class="tg-0lax">$MMOCR_DIR/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMSeg</summary>
<div style="margin-left: 25px;">
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow" colspan="2">MMSeg</th>
    <th class="tg-0lax">Pytorch</th>
    <th class="tg-0pky">ONNXRuntime</th>
    <th class="tg-c3ow" colspan="3"><span style="font-weight:400;font-style:normal">TensorRT</span></th>
    <th class="tg-c3ow">PPLNN</th>
    <th class="tg-0pky"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8">Model</td>
    <td class="tg-0pky">Metrics</td>
    <td class="tg-baqh">fp32</td>
    <td class="tg-c3ow">fp32</td>
    <td class="tg-c3ow">fp32</td>
    <td class="tg-c3ow"><span style="font-weight:400;font-style:normal">fp16</span></td>
    <td class="tg-c3ow">int8</td>
    <td class="tg-c3ow">fp16</td>
    <td class="tg-lboi">model config file</td>
  </tr>
  <tr>
    <td class="tg-0pky">FCN</td>
    <td class="tg-0pky">mIoU</td>
    <td class="tg-0lax">72.25</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">72.36</td>
    <td class="tg-c3ow">72.35</td>
    <td class="tg-c3ow">74.19</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">PSPNet</td>
    <td class="tg-0pky">mIoU</td>
    <td class="tg-0lax">78.55</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">78.26</td>
    <td class="tg-c3ow">78.24</td>
    <td class="tg-c3ow">77.97</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td class="tg-0pky">deeplabv3</td>
    <td class="tg-0pky">mIoU</td>
    <td class="tg-0lax">79.09</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">79.12</td>
    <td class="tg-c3ow">79.12</td>
    <td class="tg-c3ow">78.96</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
    <td class="tg-0pky">deeplabv3+</td>
    <td class="tg-0pky">mIoU</td>
    <td class="tg-0lax">79.61</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">79.6</td>
    <td class="tg-c3ow">79.6</td>
    <td class="tg-c3ow">79.43</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  </tr>
    <td class="tg-0pky">Fast-SCNN</td>
    <td class="tg-0pky">mIoU</td>
    <td class="tg-0lax">70.96</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">70.93</td>
    <td class="tg-c3ow">70.92</td>
    <td class="tg-c3ow">66.0</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-0pky">$MMSEG_DIR/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py</td>
  </tr>
</tbody>
</table>
</div>
</details>


### Notes
- As some datasets contains images with various resolutions in codebase like MMDet. The speed benchmark is gained through static configs in MMDeploy, while the performance benchmark is gained through dynamic ones.

- Some int8 performance benchmarks of TensorRT require nvidia cards with tensor core, or the performance would drop heavily.

- DBNet uses the interpolate mode `nearest` in the neck of the model, which TensorRT-7 applies quite different strategy from pytorch. To make the repository compatible with TensorRT-7, we rewrite the neck to use the interpolate mode `bilinear` which improves final detection performance. To get the matched performance with Pytorch, TensorRT-8+ is recommended, which the interpolate methods are all the same as Pytorch.
