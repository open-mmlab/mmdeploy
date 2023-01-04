# generate_md_table

生成mmdeploy支持的后端表。

## 用法

```shell
python tools/generate_md_table.py \
    ${yml_file} \
    ${output} \
    ${backends}
```

## 参数说明

- `yml_file:`  输入 yml 配置路径
- `output:` 输出markdown文件路径
- `backends:` 要输出的后端，默认为 onnxruntime tensorrt torchscript pplnn openvino ncnn

## 使用举例

从 mmocr.yml 生成mmdeploy支持的后端表

```shell
python tools/generate_md_table.py tests/regression/mmocr.yml tests/regression/mmocr.md onnxruntime tensorrt torchscript pplnn openvino ncnn
```

输出：

| model                                                                        | task            | onnxruntime | tensorrt | torchscript | pplnn | openvino | ncnn |
| :--------------------------------------------------------------------------- | :-------------- | :---------- | :------- | :---------- | :---- | :------- | :--- |
| [DBNet](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet) | TextDetection   | Y           | Y        | Y           | Y     | Y        | Y    |
| [CRNN](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn) | TextRecognition | Y           | Y        | Y           | Y     | N        | Y    |
| [SAR](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)   | TextRecognition | Y           | N        | N           | N     | N        | N    |
