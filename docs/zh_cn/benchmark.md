## 基准

### 后端

CPU: ncnn, ONNXRuntime, OpenVINO

GPU: ncnn, TensorRT, PPLNN

### 延迟基准

#### 平台

- Ubuntu 18.04 操作系统
- ncnn 20211208
- Cuda 11.3
- TensorRT 7.2.3.4
- Docker 20.10.8
- NVIDIA tesla T4 显卡.

#### 其他设置

- 静态图导出
- 批次大小为 1
- 每次推理后均同步
- 延迟基准测试时，我们计算各个数据集中100张图片的平均延时。
- 热身。 针对ncnn后端，我们热身30轮; 对于其他后端:针对分类任务，我们热身1010轮，对其他任务，我们热身10轮。
- 输入分辨率根据代码库的数据集不同而不同，除了`mmediting`，其他代码库均使用真实图片作为输入。

用户可以直接通过[如何测试延迟](tutorials/how_to_measure_performance_of_models.md)获得想要的速度测试结果。下面是我们环境中的测试结果：

<details>
<summary style="margin-left: 25px;">MMCls</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">MMCls</th>
    <th align="center" colspan="12">TensorRT</th>
    <th align="center" colspan="2">PPLNN</th>
    <th align="center" colspan="4">NCNN</th>
    <th></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="3">Model</td>
    <td align="center" rowspan="3">Dataset</td>
    <td align="center" rowspan="3">Input</td>
    <td align="center" colspan="6">T4</td>
    <td align="center" colspan="4">JetsonNano2GB</td>
    <td align="center" colspan="2">Jetson TX2</td>
    <td align="center" colspan="2">T4</td>
    <td align="center" colspan="2">SnapDragon888</td>
    <td align="center" colspan="2">Adreno660</td>
    <td rowspan="3">model config file</td>
  </tr>
  <tr>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">int8</td>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp32</td>
  </tr>
  <tr>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
  </tr>
  <tr>
    <td align="center">ResNet</td>
    <td align="center">ImageNet</td>
    <td align="center">1x3x224x224</td>
    <td align="center">2.97</td>
    <td align="center">336.90</td>
    <td align="center">1.26</td>
    <td align="center">791.89</td>
    <td align="center">1.21</td>
    <td align="center">829.66</td>
    <td align="center">59.32</td>
    <td align="center">16.86</td>
    <td align="center">30.54</td>
    <td align="center">32.75</td>
    <td align="center">24.13</td>
    <td align="center">41.44</td>
    <td align="center">1.30</td>
    <td align="center">768.28</td>
    <td align="center">33.91</td>
    <td align="center">29.49</td>
    <td align="center">25.93</td>
    <td align="center">38.57</td>
    <td>$MMCLS_DIR/configs/resnet/resnet50_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td align="center">ResNeXt</td>
    <td align="center">ImageNet</td>
    <td align="center">1x3x224x224</td>
    <td align="center">4.31</td>
    <td align="center">231.93</td>
    <td align="center">1.42</td>
    <td align="center">703.42</td>
    <td align="center">1.37</td>
    <td align="center">727.42</td>
    <td align="center">88.10</td>
    <td align="center">11.35</td>
    <td align="center">49.18</td>
    <td align="center">20.13</td>
    <td align="center">37.45</td>
    <td align="center">26.70</td>
    <td align="center">1.36</td>
    <td align="center">737.67</td>
    <td align="center">133.44</td>
    <td align="center">7.49</td>
    <td align="center">69.38</td>
    <td align="center">14.41</td>
    <td>$MMCLS_DIR/configs/resnext/resnext50_32x4d_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td align="center">SE-ResNet</td>
    <td align="center">ImageNet</td>
    <td align="center">1x3x224x224</td>
    <td align="center">3.41</td>
    <td align="center">293.64</td>
    <td align="center">1.66</td>
    <td align="center">600.73</td>
    <td align="center">1.51</td>
    <td align="center">662.90</td>
    <td align="center">74.59</td>
    <td align="center">13.41</td>
    <td align="center">48.78</td>
    <td align="center">20.50</td>
    <td align="center">29.62</td>
    <td align="center">33.76</td>
    <td align="center">1.91</td>
    <td align="center">524.07</td>
    <td align="center">107.84</td>
    <td align="center">9.27</td>
    <td align="center">80.85</td>
    <td align="center">12.37</td>
    <td>$MMCLS_DIR/configs/seresnet/seresnet50_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td align="center">ShuffleNetV2</td>
    <td align="center">ImageNet</td>
    <td align="center">1x3x224x224</td>
    <td align="center">1.37</td>
    <td align="center">727.94</td>
    <td align="center">1.19</td>
    <td align="center">841.36</td>
    <td align="center">1.13</td>
    <td align="center">883.47</td>
    <td align="center">15.26</td>
    <td align="center">65.54</td>
    <td align="center">10.23</td>
    <td align="center">97.77</td>
    <td align="center">7.37</td>
    <td align="center">135.73</td>
    <td align="center">4.69</td>
    <td align="center">213.33</td>
    <td align="center">9.55</td>
    <td align="center">104.71</td>
    <td align="center">10.66</td>
    <td align="center">93.81</td>
    <td>$MMCLS_DIR/configs/shufflenet_v2/shufflenet_v2_1x_b64x16_linearlr_bn_nowd_imagenet.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMDet</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">MMDet</th>
    <th align="center" colspan="8">TensorRT</th>
    <th align="center" colspan="2">PPLNN</th>
    <th></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="3">Model</td>
    <td align="center" rowspan="3">Dataset</td>
    <td align="center" rowspan="3">Input</td>
    <td align="center" colspan="6">T4</td>
    <td align="center" colspan="2">Jetson TX2</td>
    <td align="center" colspan="2">T4</td>
    <td rowspan="3">model config file</td>
  </tr>
  <tr>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">int8</td>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
  </tr>
  <tr>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
  </tr>
  <tr>
    <td align="center">YOLOv3</td>
    <td align="center">COCO</td>
    <td align="center">1x3x320x320</td>
    <td align="center">14.76</td>
    <td align="center">67.76</td>
    <td align="center">24.92</td>
    <td align="center">40.13</td>
    <td align="center">24.92</td>
    <td align="center">40.13</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">18.07</td>
    <td align="center">55.35</td>
    <td>$MMDET_DIR/configs/yolo/yolov3_d53_320_273e_coco.py</td>
  </tr>
  <tr>
    <td align="center">SSD-Lite</td>
    <td align="center">COCO</td>
    <td align="center">1x3x320x320</td>
    <td align="center">8.84</td>
    <td align="center">113.12</td>
    <td align="center">9.21</td>
    <td align="center">108.56</td>
    <td align="center">8.04</td>
    <td align="center">124.38</td>
    <td align="center">1.28</td>
    <td align="center">1.28</td>
    <td align="center">19.72</td>
    <td align="center">50.71</td>
    <td>$MMDET_DIR/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py</td>
  </tr>
  <tr>
    <td align="center">RetinaNet</td>
    <td align="center">COCO</td>
    <td align="center">1x3x800x1344</td>
    <td align="center">97.09</td>
    <td align="center">10.30</td>
    <td align="center">25.79</td>
    <td align="center">38.78</td>
    <td align="center">16.88</td>
    <td align="center">59.23</td>
    <td align="center">780.48</td>
    <td align="center">1.28</td>
    <td align="center">38.34</td>
    <td align="center">26.08</td>
    <td>$MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">FCOS</td>
    <td align="center">COCO</td>
    <td align="center">1x3x800x1344</td>
    <td align="center">84.06</td>
    <td align="center">11.90</td>
    <td align="center">23.15</td>
    <td align="center">43.20</td>
    <td align="center">17.68</td>
    <td align="center">56.57</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">FSAF</td>
    <td align="center">COCO</td>
    <td align="center">1x3x800x1344</td>
    <td align="center">82.96</td>
    <td align="center">12.05</td>
    <td align="center">21.02</td>
    <td align="center">47.58</td>
    <td align="center">13.50</td>
    <td align="center">74.08</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">30.41</td>
    <td align="center">32.89</td>
    <td>$MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">Faster-RCNN</td>
    <td align="center">COCO</td>
    <td align="center">1x3x800x1344</td>
    <td align="center">88.08</td>
    <td align="center">11.35</td>
    <td align="center">26.52</td>
    <td align="center">37.70</td>
    <td align="center">19.14</td>
    <td align="center">52.23</td>
    <td align="center">733.81</td>
    <td align="center">1.36</td>
    <td align="center">65.40</td>
    <td align="center">15.29</td>
    <td>$MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">Mask-RCNN</td>
    <td align="center">COCO</td>
    <td align="center">1x3x800x1344</td>
    <td align="center">104.83</td>
    <td align="center">9.54</td>
    <td align="center">58.27</td>
    <td align="center">17.16</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">86.80</td>
    <td align="center">11.52</td>
    <td>$MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
</tbody>
</table>
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">MMDet</th>
    <th align="center" colspan="4">NCNN</th>
    <th align="center"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="3">Model</td>
    <td align="center" rowspan="3">Dataset</td>
    <td align="center" rowspan="3">Input</td>
    <td align="center" colspan="2">SnapDragon888</td>
    <td align="center" colspan="2">Adreno660</td>
    <td rowspan="3">model config file</td>
  </tr>
  <tr>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp32</td>
  </tr>
  <tr>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
  </tr>
  <tr>
    <td align="center">MobileNetv2-YOLOv3</td>
    <td align="center">COCO</td>
    <td align="center">1x3x320x320</td>
    <td align="center">48.57</td>
    <td align="center">20.59</td>
    <td align="center">66.55</td>
    <td align="center">15.03</td>
    <td>$MMDET_DIR/configs/yolo/yolov3_mobilenetv2_mstrain-416_300e_coco.py</td>
  </tr>
  <tr>
    <td align="center">SSD-Lite</td>
    <td align="center">COCO</td>
    <td align="center">1x3x320x320</td>
    <td align="center">44.91</td>
    <td align="center">22.27</td>
    <td align="center">66.19</td>
    <td align="center">15.11</td>
    <td>$MMDET_DIR/configs/ssd/ssdlite_mobilenetv2_scratch_600e_coco.py</td>
  </tr>
  <tr>
    <td align="center">YOLOX</td>
    <td align="center">COCO</td>
    <td align="center">1x3x416x416</td>
    <td align="center">111.60</td>
    <td align="center">8.96</td>
    <td align="center">134.50</td>
    <td align="center">7.43</td>
    <td>$MMDET_DIR/configs/yolox/yolox_tiny_8x8_300e_coco.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMEdit</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="2">MMEdit</th>
    <th align="center" colspan="8">TensorRT</th>
    <th align="center" colspan="2">PPLNN</th>
    <th></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="3">Model</td>
    <td align="center" rowspan="3">Input</td>
    <td align="center" colspan="6">T4</td>
    <td align="center" colspan="2">Jetson TX2</td>
    <td align="center" colspan="2">T4</td>
    <td rowspan="3">model config file</td>
  </tr>
  <tr>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">int8</td>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
  </tr>
  <tr>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
  </tr>
  <tr>
    <td align="center">ESRGAN</td>
    <td align="center">1x3x32x32</td>
    <td align="center">12.64</td>
    <td align="center">79.14</td>
    <td align="center">12.42</td>
    <td align="center">80.50</td>
    <td align="center">12.45</td>
    <td align="center">80.35</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">7.67</td>
    <td align="center">130.39</td>
    <td>$MMEDIT_DIR/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td align="center">SRCNN</td>
    <td align="center">1x3x32x32</td>
    <td align="center">0.70</td>
    <td align="center">1436.47</td>
    <td align="center">0.35</td>
    <td align="center">2836.62</td>
    <td align="center">0.26</td>
    <td align="center">3850.45</td>
    <td align="center">58.86</td>
    <td align="center">16.99</td>
    <td align="center">0.56</td>
    <td align="center">1775.11</td>
    <td>$MMEDIT_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMOCR</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">MMOCR</th>
    <th align="center" colspan="6">TensorRT</th>
    <th align="center" colspan="2">PPLNN</th>
    <th align="center" colspan="4">NCNN</th>
    <th align="center"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="3">Model</td>
    <td align="center" rowspan="3">Dataset</td>
    <td align="center" rowspan="3">Input</td>
    <td align="center" colspan="6">T4</td>
    <td align="center" colspan="2">T4</td>
    <td align="center" colspan="2">SnapDragon888</td>
    <td align="center" colspan="2">Adreno660</td>
    <td rowspan="3">model config file</td>
  </tr>
  <tr>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">int8</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp32</td>
  </tr>
  <tr>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
  </tr>
    <tr>
    <td align="center">DBNet</td>
    <td align="center">ICDAR2015</td>
    <td align="center">1x3x640x640</td>
    <td align="center">10.70</td>
    <td align="center">93.43</td>
    <td align="center">5.62</td>
    <td align="center">177.78</td>
    <td align="center">5.00</td>
    <td align="center">199.85</td>
    <td align="center">34.84</td>
    <td align="center">28.70</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMOCR_DIR/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py</td>
  </tr>
  <tr>
    <td align="center">CRNN</td>
    <td align="center">IIIT5K</td>
    <td align="center">1x1x32x32</td>
    <td align="center">1.93 </td>
    <td align="center">518.28</td>
    <td align="center">1.40</td>
    <td align="center">713.88</td>
    <td align="center">1.36</td>
    <td align="center">736.79</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">10.57</td>
    <td align="center">94.64</td>
    <td align="center">20.00</td>
    <td align="center">50.00</td>
    <td>$MMOCR_DIR/configs/textrecog/crnn/crnn_academic_dataset.py</td>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMSeg</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">MMSeg</th>
    <th align="center" colspan="8">TensorRT</th>
    <th align="center" colspan="2">PPLNN</th>
    <th></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center" rowspan="3">Model</td>
    <td align="center" rowspan="3">Dataset</td>
    <td align="center" rowspan="3">Input</td>
    <td align="center" colspan="6">T4</td>
    <td align="center" colspan="2">Jetson TX2</td>
    <td align="center" colspan="2">T4</td>
    <td rowspan="3">model config file</td>
  </tr>
  <tr>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
    <td align="center" colspan="2">int8</td>
    <td align="center" colspan="2">fp32</td>
    <td align="center" colspan="2">fp16</td>
  </tr>
  <tr>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
    <td align="center">latency (ms)</td>
    <td align="center">FPS</td>
  </tr>
  <tr>
    <td align="center">FCN</td>
    <td align="center">Cityscapes</td>
    <td align="center">1x3x512x1024</td>
    <td align="center">128.42</td>
    <td align="center">7.79</td>
    <td align="center">23.97</td>
    <td align="center">41.72</td>
    <td align="center">18.13</td>
    <td align="center">55.15</td>
    <td align="center">1682.54</td>
    <td align="center">0.59</td>
    <td align="center">27.00</td>
    <td align="center">37.04</td>
    <td>$MMSEG_DIR/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">PSPNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">1x3x512x1024</td>
    <td align="center">119.77</td>
    <td align="center">8.35</td>
    <td align="center">24.10</td>
    <td align="center">41.49</td>
    <td align="center">16.33</td>
    <td align="center">61.23</td>
    <td align="center">1586.19</td>
    <td align="center">0.63</td>
    <td align="center">27.26</td>
    <td align="center">36.69</td>
    <td>$MMSEG_DIR/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">DeepLabV3</td>
    <td align="center">Cityscapes</td>
    <td align="center">1x3x512x1024</td>
    <td align="center">226.75</td>
    <td align="center">4.41</td>
    <td align="center">31.80</td>
    <td align="center">31.45</td>
    <td align="center">19.85</td>
    <td align="center">50.38</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">36.01</td>
    <td align="center">27.77</td>
    <td>$MMSEG_DIR/configs/deeplabv3/deeplabv3_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">DeepLabV3+</td>
    <td align="center">Cityscapes</td>
    <td align="center">1x3x512x1024</td>
    <td align="center">151.25</td>
    <td align="center">6.61</td>
    <td align="center">47.03</td>
    <td align="center">21.26</td>
    <td align="center">50.38</td>
    <td align="center">26.67</td>
    <td align="center">2534.96</td>
    <td align="center">0.39</td>
    <td align="center">34.80</td>
    <td align="center">28.74</td>
    <td>$MMSEG_DIR/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

### 性能基准

用户可以直接通过[如何测试性能](tutorials/how_to_evaluate_a_model.md)获得想要的性能测试结果。下面是我们环境中的测试结果：


<details>
<summary style="margin-left: 25px;">MMCls</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">MMCls</th>
    <th align="center">PyTorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNX Runtime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Model</td>
    <td align="center">Task</td>
    <td align="center">Metrics</td>
    <td align="center">fp32</td>
    <td align="center">seresnet</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
    <td>model config file</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">ResNet-18</td>
    <td align="center" rowspan="2">Classification</td>
    <td align="center">top-1</td>
    <td align="center">69.90</td>
    <td align="center">69.90</td>
    <td align="center">69.88</td>
    <td align="center">69.88</td>
    <td align="center">69.86</td>
    <td align="center">69.86</td>
    <td align="center">69.86</td>
    <td rowspan="2">$MMCLS_DIR/configs/resnet/resnet18_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">89.43</td>
    <td align="center">89.43</td>
    <td align="center">89.34</td>
    <td align="center">89.34</td>
    <td align="center">89.33</td>
    <td align="center">89.38</td>
    <td align="center">89.34</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">ResNeXt-50</td>
    <td align="center" rowspan="2">Classification</td>
    <td align="center">top-1</td>
    <td align="center">77.90</td>
    <td align="center">77.90</td>
    <td align="center">77.90</td>
    <td align="center">77.90</td>
    <td align="center">-</td>
    <td align="center">77.78</td>
    <td align="center">77.89</td>
    <td rowspan="2">$MMCLS_DIR/configs/resnext/resnext50_32x4d_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">93.66</td>
    <td align="center">93.66</td>
    <td align="center">93.66</td>
    <td align="center">93.66</td>
    <td align="center">-</td>
    <td align="center">93.64</td>
    <td align="center">93.65</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">SE-ResNet-50</td>
    <td align="center" rowspan="2">Classification</td>
    <td align="center">top-1</td>
    <td align="center">77.74</td>
    <td align="center">77.74</td>
    <td align="center">77.74</td>
    <td align="center">77.74</td>
    <td align="center">77.75</td>
    <td align="center">77.63</td>
    <td align="center">77.73</td>
    <td rowspan="2">$MMCLS_DIR/configs/resnext/resnext50_32x4d_b32x8_imagenet.py</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">93.84</td>
    <td align="center">93.84</td>
    <td align="center">93.84</td>
    <td align="center">93.84</td>
    <td align="center">93.83</td>
    <td align="center">93.72</td>
    <td align="center">93.84</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">ShuffleNetV1</td>
    <td align="center" rowspan="2">Classification</td>
    <td align="center">top-1</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">68.13</td>
    <td align="center">67.71</td>
    <td align="center">68.11</td>
    <td rowspan="2">$MMCLS_DIR/configs/shufflenet_v1/shufflenet-v1-1x_16xb64_in1k.py</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.81</td>
    <td align="center">87.58</td>
    <td align="center">87.80</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">ShuffleNetV2</td>
    <td align="center" rowspan="2">Classification</td>
    <td align="center">top-1</td>
    <td align="center">69.55</td>
    <td align="center">69.55</td>
    <td align="center">69.55</td>
    <td align="center">69.55</td>
    <td align="center">69.54</td>
    <td align="center">69.10</td>
    <td align="center">69.54</td>
    <td rowspan="2">$MMCLS_DIR/configs/shufflenet_v2/shufflenet-v2-1x_16xb64_in1k.py</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">88.92</td>
    <td align="center">88.92</td>
    <td align="center">88.92</td>
    <td align="center">88.92</td>
    <td align="center">88.91</td>
    <td align="center">88.58</td>
    <td align="center">88.92</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">MobileNet V2</td>
    <td align="center" rowspan="2">Classification</td>
    <td align="center">top-1</td>
    <td align="center">71.86</td>
    <td align="center">71.86</td>
    <td align="center">71.86</td>
    <td align="center">71.86</td>
    <td align="center">71.87</td>
    <td align="center">70.91</td>
    <td align="center">71.84</td>
    <td rowspan="2">$MMEDIT_DIR/configs/mobilenet_v2/mobilenet-v2_8xb32_in1k.py</td>
  </tr>
  <tr>
    <td align="center">top-5</td>
    <td align="center">90.42</td>
    <td align="center">90.42</td>
    <td align="center">90.42</td>
    <td align="center">90.42</td>
    <td align="center">90.40</td>
    <td align="center">89.85</td>
    <td align="center">90.41</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMDet</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">MMDet</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center">OpenVINO</th>
    <th align="center"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Model</td>
    <td align="center">Task</td>
    <td align="center">Dataset</td>
    <td align="center">Metrics</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
    <td align="center">fp32</td>
    <td>model config file</td>
  </tr>
  <tr>
    <td align="center">YOLOV3</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">33.7</td>
    <td align="center">33.7</td>
    <td align="center">-</td>
    <td align="center">33.5</td>
    <td align="center">33.5</td>
    <td align="center">33.5</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/yolo/yolov3_d53_320_273e_coco.py</td>
  </tr>
  <tr>
    <td align="center">SSD</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">25.5</td>
    <td align="center">25.5</td>
    <td align="center">-</td>
    <td align="center">25.5</td>
    <td align="center">25.5</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/ssd/ssd300_coco.py</td>
  </tr>
  <tr>
    <td align="center">RetinaNet</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">36.5</td>
    <td align="center">36.4</td>
    <td align="center">-</td>
    <td align="center">36.4</td>
    <td align="center">36.4</td>
    <td align="center">36.3</td>
    <td align="center">36.5</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/retinanet/retinanet_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">FCOS</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">36.6</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">36.6</td>
    <td align="center">36.5</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/fcos/fcos_r50_caffe_fpn_gn-head_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">FSAF</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">37.4</td>
    <td align="center">37.4</td>
    <td align="center">-</td>
    <td align="center">37.4</td>
    <td align="center">37.4</td>
    <td align="center">37.2</td>
    <td align="center">37.4</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/fsaf/fsaf_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">YOLOX</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">40.5</td>
    <td align="center">40.3</td>
    <td align="center">-</td>
    <td align="center">40.3</td>
    <td align="center">40.3</td>
    <td align="center">29.3</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/yolox/yolox_s_8x8_300e_coco.py</td>
  </tr>
  <tr>
    <td align="center">Faster R-CNN</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">37.4</td>
    <td align="center">37.3</td>
    <td align="center">-</td>
    <td align="center">37.3</td>
    <td align="center">37.3</td>
    <td align="center">37.1</td>
    <td align="center">37.3</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">ATSS</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">39.4</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">39.4</td>
    <td align="center">39.4</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/atss/atss_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">Cascade R-CNN</td>
    <td align="center">Object Detection</td>
    <td align="center">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">40.4</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">40.4</td>
    <td align="center">40.4</td>
    <td align="center">-</td>
    <td align="center">40.4</td>
    <td align="center">-</td>
    <td>$MMDET_DIR/configs/cascade_rcnn/cascade_rcnn_r50_caffe_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">Mask R-CNN</td>
    <td align="center" rowspan="2">Instance Segmentation</td>
    <td align="center" rowspan="2">COCO2017</td>
    <td align="center">box AP</td>
    <td align="center">38.2</td>
    <td align="center">38.1</td>
    <td align="center">-</td>
    <td align="center">38.1</td>
    <td align="center">38.1</td>
    <td align="center">-</td>
    <td align="center">38.0</td>
    <td align="center">-</td>
    <td rowspan="2">$MMDET_DIR/configs/mask_rcnn/mask_rcnn_r50_fpn_1x_coco.py</td>
  </tr>
  <tr>
    <td align="center">mask AP</td>
    <td align="center">34.7</td>
    <td align="center">34.7</td>
    <td align="center">-</td>
    <td align="center">33.7</td>
    <td align="center">33.7</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMEdit</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">MMEdit</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNX Runtime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Model</td>
    <td align="center">Task</td>
    <td align="center">Dataset</td>
    <td align="center">Metrics</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
    <td>model config file</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">SRCNN</td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">28.4316</td>
    <td align="center">28.4120</td>
    <td align="center">28.4323</td>
    <td align="center">28.4323</td>
    <td align="center">28.4286</td>
    <td align="center">28.1995</td>
    <td align="center">28.4311</td>
    <td rowspan="2">$MMEDIT_DIR/configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8099</td>
    <td align="center">0.8106</td>
    <td align="center">0.8097</td>
    <td align="center">0.8097</td>
    <td align="center">0.8096</td>
    <td align="center">0.7934</td>
    <td align="center">0.8096</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">ESRGAN</td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">28.2700</td>
    <td align="center">28.2619</td>
    <td align="center">28.2592</td>
    <td align="center">28.2592</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">28.2624</td>
    <td rowspan="2">$MMEDIT_DIR/configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.7778</td>
    <td align="center">0.7784</td>
    <td align="center">0.7764</td>
    <td align="center">0.7774</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.7765</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">ESRGAN-PSNR</td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">30.6428</td>
    <td align="center">30.6306</td>
    <td align="center">30.6444</td>
    <td align="center">30.6430</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">27.0426</td>
    <td rowspan="2">$MMEDIT_DIR/configs/restorers/esrgan/esrgan_psnr_x4c64b23g32_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8559</td>
    <td align="center">0.8565</td>
    <td align="center">0.8558</td>
    <td align="center">0.8558</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.8557</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">SRGAN</td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">27.9499</td>
    <td align="center">27.9252</td>
    <td align="center">27.9408</td>
    <td align="center">27.9408</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">27.9388</td>
    <td rowspan="2">$MMEDIT_DIR/configs/restorers/srresnet_srgan/srgan_x4c64b16_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.7846</td>
    <td align="center">0.7851</td>
    <td align="center">0.7839</td>
    <td align="center">0.7839</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.7839</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">SRResNet</td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">30.2252</td>
    <td align="center">30.2069</td>
    <td align="center">30.2300</td>
    <td align="center">30.2300</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">30.2294</td>
    <td rowspan="2">$MMEDIT_DIR/configs/restorers/srresnet_srgan/msrresnet_x4c64b16_g1_1000k_div2k.py</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8491</td>
    <td align="center">0.8497</td>
    <td align="center">0.8488</td>
    <td align="center">0.8488</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.8488</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">Real-ESRNet</td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">28.0297</td>
    <td align="center">-</td>
    <td align="center">27.7016</td>
    <td align="center">27.7016</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">27.7049</td>
    <td rowspan="2">$MMEDIT_DIR/configs/restorers/real_esrgan/realesrnet_c64b23g32_12x4_lr2e-4_1000k_df2k_ost.py</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8236</td>
    <td align="center">-</td>
    <td align="center">0.8122</td>
    <td align="center">0.8122</td>
    <td align="center"> - </td>
    <td align="center"> - </td>
    <td align="center">0.8123</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">EDSR</td>
    <td align="center" rowspan="2">Super Resolution</td>
    <td align="center" rowspan="2">Set5</td>
    <td align="center">PSNR</td>
    <td align="center">30.2223</td>
    <td align="center">30.2192</td>
    <td align="center">30.2214</td>
    <td align="center">30.2214</td>
    <td align="center">30.2211</td>
    <td align="center">30.1383</td>
    <td align="center">-</td>
    <td rowspan="2">$MMEDIT_DIR/configs/restorers/edsr/edsr_x4c64b16_g1_300k_div2k.py</td>
  </tr>
  <tr>
    <td align="center">SSIM</td>
    <td align="center">0.8500</td>
    <td align="center">0.8507</td>
    <td align="center">0.8497</td>
    <td align="center">0.8497</td>
    <td align="center">0.8497</td>
    <td align="center">0.8469</td>
    <td align="center"> - </td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMOCR</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">MMOCR</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center">OpenVINO</th>
    <th align="center"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Model</td>
    <td align="center">Task</td>
    <td align="center">Dataset</td>
    <td align="center">Metrics</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
    <td align="center">fp32</td>
    <td>model config file</td>
  </tr>
  <tr>
    <td align="center" rowspan="3">DBNet*</td>
    <td align="center" rowspan="3">TextDetection</td>
    <td align="center" rowspan="3">ICDAR2015</td>
    <td align="center">recall</td>
    <td align="center">0.7310</td>
    <td align="center">0.7308</td>
    <td align="center">0.7304</td>
    <td align="center">0.7198</td>
    <td align="center">0.7179</td>
    <td align="center">0.7111</td>
    <td align="center">0.7304</td>
    <td align="center">0.7309</td>
    <td align="center" rowspan="3">$MMOCR_DIR/configs/textdet/dbnet/dbnet_r18_fpnc_1200e_icdar2015.py</td>
  </tr>
  <tr>
    <td align="center">precision</td>
    <td align="center">0.8714</td>
    <td align="center">0.8718</td>
    <td align="center">0.8714</td>
    <td align="center">0.8677</td>
    <td align="center">0.8674</td>
    <td align="center">0.8688</td>
    <td align="center">0.8718</td>
    <td align="center">0.8714</td>
  </tr>
  <tr>
    <td align="center">hmean</td>
    <td align="center">0.7950</td>
    <td align="center">0.7949</td>
    <td align="center">0.7950</td>
    <td align="center">0.7868</td>
    <td align="center">0.7856</td>
    <td align="center">0.7821</td>
    <td align="center">0.7949</td>
    <td align="center">0.7950</td>
  </tr>
  <tr>
    <td align="center">CRNN</td>
    <td align="center">TextRecognition</td>
    <td align="center">IIIT5K</td>
    <td align="center">acc</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">0.8063</td>
    <td align="center">0.8067</td>
    <td align="center">0.8067</td>
    <td align="center">-</td>
    <td>$MMOCR_DIR/configs/textrecog/crnn/crnn_academic_dataset.py</td>
  </tr>
  <tr>
    <td align="center">SAR</td>
    <td align="center">TextRecognition</td>
    <td align="center">IIIT5K</td>
    <td align="center">acc</td>
    <td align="center">0.9517</td>
    <td align="center">-</td>
    <td align="center">0.9287</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMOCR_DIR/configs/textrecog/sar/sar_r31_parallel_decoder_academic.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMSeg</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="3">MMSeg</th>
    <th align="center">Pytorch</th>
    <th align="center">TorchScript</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="3">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center"></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Model</td>
    <td align="center">Dataset</td>
    <td align="center">Metrics</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">int8</td>
    <td align="center">fp16</td>
    <td>model config file</td>
  </tr>
  <tr>
    <td align="center">FCN</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">72.25</td>
    <td align="center">72.36</td>
    <td align="center">-</td>
    <td align="center">72.36</td>
    <td align="center">72.35</td>
    <td align="center">74.19</td>
    <td align="center">72.35</td>
    <td>$MMSEG_DIR/configs/fcn/fcn_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">PSPNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">78.55</td>
    <td align="center">78.66</td>
    <td align="center">-</td>
    <td align="center">78.26</td>
    <td align="center">78.24</td>
    <td align="center">77.97</td>
    <td align="center">78.09</td>
    <td>$MMSEG_DIR/configs/pspnet/pspnet_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">deeplabv3</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">79.09</td>
    <td align="center">79.12</td>
    <td align="center">-</td>
    <td align="center">79.12</td>
    <td align="center">79.12</td>
    <td align="center">78.96</td>
    <td align="center">79.12</td>
    <td>$MMSEG_DIR/configs/deeplabv3/deeplabv3_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">deeplabv3+</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">79.61</td>
    <td align="center">79.60</td>
    <td align="center">-</td>
    <td align="center">79.60</td>
    <td align="center">79.60</td>
    <td align="center">79.43</td>
    <td align="center">79.60</td>
    <td>$MMSEG_DIR/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">Fast-SCNN</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">70.96</td>
    <td align="center">70.96</td>
    <td align="center">-</td>
    <td align="center">70.93</td>
    <td align="center">70.92</td>
    <td align="center">66.00</td>
    <td align="center">70.92</td>
    <td>$MMSEG_DIR/configs/fastscnn/fast_scnn_lr0.12_8x4_160k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">UNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">69.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">69.10</td>
    <td align="center">69.10</td>
    <td align="center">68.95</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/unet/fcn_unet_s5-d16_4x4_512x1024_160k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">ANN</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.40</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.32</td>
    <td align="center">77.32</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/ann/ann_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">APCNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.40</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.32</td>
    <td align="center">77.32</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/apcnet/apcnet_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">BiSeNetV1</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">74.44</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">74.44</td>
    <td align="center">74.43</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/bisenetv1/bisenetv1_r18-d32_4x4_1024x1024_160k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">BiSeNetV2</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">73.21</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">73.21</td>
    <td align="center">73.21</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/bisenetv2/bisenetv2_fcn_4x4_1024x1024_160k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">CGNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">68.25</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">68.27</td>
    <td align="center">68.27</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/cgnet/cgnet_512x1024_60k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">EMANet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.59</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.59</td>
    <td align="center">77.6</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/emanet/emanet_r50-d8_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">EncNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">75.67</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">75.66</td>
    <td align="center">75.66</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/encnet/encnet_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">ERFNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">71.08</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">71.08</td>
    <td align="center">71.07</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/erfnet/erfnet_fcn_4x4_512x1024_160k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">FastFCN</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">79.12</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">79.12</td>
    <td align="center">79.12</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/fastfcn/fastfcn_r50-d32_jpu_aspp_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">GCNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.69</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.69</td>
    <td align="center">77.69</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/gcnet/gcnet_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">ICNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">76.29</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">76.36</td>
    <td align="center">76.36</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/icnet/icnet_r18-d8_832x832_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">ISANet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">78.49</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">78.49</td>
    <td align="center">78.49</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/isanet/isanet_r50-d8_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">OCRNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">74.30</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">73.66</td>
    <td align="center">73.67</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/ocrnet/ocrnet_hr18s_512x1024_40k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">PointRend</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">76.47</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">76.41</td>
    <td align="center">76.42</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/point_rend/pointrend_r50_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">Semantic FPN</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">74.52</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">74.52</td>
    <td align="center">74.52</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/sem_fpn/fpn_r50_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">STDC</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">75.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">75.10</td>
    <td align="center">75.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/stdc/stdc1_in1k-pre_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">STDC</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.17</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.17</td>
    <td align="center">77.17</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/stdc/stdc2_in1k-pre_512x1024_80k_cityscapes.py</td>
  </tr>
  <tr>
    <td align="center">UPerNet</td>
    <td align="center">Cityscapes</td>
    <td align="center">mIoU</td>
    <td align="center">77.10</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">77.19</td>
    <td align="center">77.18</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td>$MMSEG_DIR/configs/upernet/upernet_r50_512x1024_40k_cityscapes.py</td>
  </tr>
</tbody>
</table>
</div>
</details>

<details>
<summary style="margin-left: 25px;">MMPose</summary>
<div style="margin-left: 25px;">
<table class="docutils">
<thead>
  <tr>
    <th align="center" colspan="4">MMpose</th>
    <th align="center">Pytorch</th>
    <th align="center">ONNXRuntime</th>
    <th align="center" colspan="2">TensorRT</th>
    <th align="center">PPLNN</th>
    <th align="center">OpenVINO</th>
    <th align="left">Model Config</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center">Model</td>
    <td align="center">Task</td>
    <td align="center">Dataset</td>
    <td align="center">Metrics</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp32</td>
    <td align="center">fp16</td>
    <td align="center">fp16</td>
    <td align="center">fp32</td>
    <td>model config file</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">HRNet</td>
    <td align="center" rowspan="2">Pose Detection</td>
    <td align="center" rowspan="2">COCO</td>
    <td align="center">AP</td>
    <td align="center">0.748</td>
    <td align="center">0.748</td>
    <td align="center">0.748</td>
    <td align="center">0.748</td>
    <td align="center">-</td>
    <td align="center">0.748</td>
    <td rowspan="2">$MMPOSE_DIR/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/hrnet_w48_coco_256x192.py</td>
  </tr>
  <tr>
    <td align="center">AR</td>
    <td align="center">0.802</td>
    <td align="center">0.802</td>
    <td align="center">0.802</td>
    <td align="center">0.802</td>
    <td align="center">-</td>
    <td align="center">0.802</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">LiteHRNet</td>
    <td align="center" rowspan="2">Pose Detection</td>
    <td align="center" rowspan="2">COCO</td>
    <td align="center">AP</td>
    <td align="center">0.663</td>
    <td align="center">0.663</td>
    <td align="center">0.663</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.663</td>
    <td rowspan="2">$MMPOSE_DIR/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/litehrnet_30_coco_256x192.py</td>
  </tr>
  <tr>
    <td align="center">AR</td>
    <td align="center">0.728</td>
    <td align="center">0.728</td>
    <td align="center">0.728</td>
    <td align="center">-</td>
    <td align="center">-</td>
    <td align="center">0.728</td>
  </tr>
  <tr>
    <td align="center" rowspan="2">MSPN </td>
    <td align="center" rowspan="2">Pose Detection</td>
    <td align="center" rowspan="2">COCO</td>
    <td align="center">AP</td>
    <td align="center">0.762</td>
    <td align="center">0.762</td>
    <td align="center">0.762</td>
    <td align="center">0.762</td>
    <td align="center">-</td>
    <td align="center">0.762</td>
    <td rowspan="2">$MMPOSE_DIR/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/4xmspn50_coco_256x192.py</td>
  </tr>
  <tr>
    <td align="center">AR</td>
    <td align="center">0.825</td>
    <td align="center">0.825</td>
    <td align="center">0.825</td>
    <td align="center">0.825</td>
    <td align="center">-</td>
    <td align="center">0.825</td>
  </tr>
</tbody>
</table>
</div>
</details>

### 注意

- 由于某些数据集在代码库中包含各种分辨率的图像，例如 MMDet，速度基准是通过 MMDeploy 中的静态配置获得的，而性能基准是通过动态配置获得的。

- TensorRT 的一些 int8 性能基准测试需要具有 tensor core 的 Nvidia 卡，否则性能会大幅下降。

- DBNet 在模型的颈部使用了`nearest`插值模式，TensorRT-7 应用了与 Pytorch 完全不同的策略。为了使与 TensorRT-7 兼容，我们重写了`neck`以使用`bilinear`插值模式，这提高了最终检测性能。为了获得与 Pytorch 匹配的性能，推荐使用 TensorRT-8+，其插值方法与 Pytorch 相同。

- MMPose 中的模型是在模型配置文件中 `flip_test` 设置为 `False`条件下完成的。
