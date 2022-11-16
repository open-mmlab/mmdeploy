# How to generate supported-backends markdown table

This tutorial describes how to generate supported-backends markdown table.

## 1.python Environment

```
pip install easydict
```

## 2.Usage

```
python tests/regression/generate_md_table.py \
    ${yml_file} \
    ${output}
```

### Description

- `yml_file:` input yml config path
- `output:`  output markdown file path

### Example

Generate backends markdown table from mmseg.yml

```
python tests/regression/generate_md_table.py tests/regression/mmseg.yml tests/regression/mmseg.md
```

## 3.Generated Table

This is an example of MMOCR generate backends markdown table

| model                                                                        | task            | onnxruntime | tensorrt | torchscript | pplnn | openvino | ncnn |
| :--------------------------------------------------------------------------- | :-------------- | :---------- | :------- | :---------- | :---- | :------- | :--- |
| [DBNet](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet) | TextDetection   | Y           | Y        | Y           | Y     | Y        | Y    |
| [CRNN](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn) | TextRecognition | Y           | Y        | Y           | Y     | N        | Y    |
| [SAR](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)   | TextRecognition | Y           | N        | N           | N     | N        | N    |
