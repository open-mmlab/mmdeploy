# 如何生成mmdeploy支持的后端表

本教程介绍如何生成mmdeploy支持的后端表。

## 1.Python 环境依赖

需要安装generate_md_table的环境

```
pip install easydict
```

## 2.用法

```
python tests/regression/generate_md_table.py \
    ${yml_file} \
    ${output}
```

### 参数解析

- `yml_file:`  输入 yml 配置路径
- `output:` 输出markdown文件路径

### 例子

从 mmseg.yml 生成mmdeploy支持的后端表

```
python tests/regression/generate_md_table.py tests/regression/mmseg.yml tests/regression/mmseg.md
```

## 3.生成的后端表

这是 MMOCR 生成的后端表

| model                                                                        | task            | onnxruntime | tensorrt | torchscript | pplnn | openvino | ncnn |
| :--------------------------------------------------------------------------- | :-------------- | :---------- | :------- | :---------- | :---- | :------- | :--- |
| [DBNet](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet) | TextDetection   | Y           | Y        | Y           | Y     | Y        | Y    |
| [CRNN](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn) | TextRecognition | Y           | Y        | Y           | Y     | N        | Y    |
| [SAR](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)   | TextRecognition | Y           | N        | N           | N     | N        | N    |
