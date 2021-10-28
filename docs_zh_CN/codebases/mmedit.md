# MMEditing 支持

[MMEditing](https://github.com/open-mmlab/mmediting) 是一个基于 PyTorch 的开源图像视频编辑工具箱。它是 [OpenMMLab](https://openmmlab.com/) 项目的一部分。

## MMEditing 安装教程

请参考[官方安装指导](https://mmediting.readthedocs.io/en/latest/install.html#installation)进代码库的安装。

## MMDeploy 支持的 MMEditing 模型列表

|  模型  |                     模型配置文件（示例）                     | ONNX Runtime | TensorRT | NCNN  |  PPL  |
| :----: | :----------------------------------------------------------: | :----------: | :------: | :---: | :---: |
| SRCNN  |    configs/restorers/srcnn/srcnn_x4k915_g1_1000k_div2k.py    |      Y       |    Y     |   N   |   Y   |
| ESRGAN | configs/restorers/esrgan/esrgan_x4c64b23g32_g1_400k_div2k.py |      Y       |    Y     |   N   |   Y   |

## MMEditing 的部署任务类型

| codebase |       task       |
| :------: | :--------------: |
|  mmedit  | super-resolution |

## 注意事项

无

## 常见问题解答

1. 为什么 SRCNN 模型在 TensorRT 上运行的精度低于在 PyTorch 上运行的精度?

    SRCNN 使用双三次插值（bicubic）来进行图像上采样。 TensorRT 不支持双三次差值操作。我们用双线性插值（bilinear）替换了该操作，这种替换会降低精度。
