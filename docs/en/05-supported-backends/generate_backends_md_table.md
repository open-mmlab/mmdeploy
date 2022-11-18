# generate_md_table

This tool can be used to generate supported-backends markdown table.

## python Environment

```
pip install easydict
```

## Usage

```
python tests/regression/generate_md_table.py \
    ${yml_file} \
    ${output}
```

## Description of all arguments

- `yml_file:` input yml config path
- `output:`  output markdown file path

## Example:

Generate backends markdown table from mmocr.yml

```
python tests/regression/generate_md_table.py tests/regression/mmocr.yml tests/regression/mmocr.md
```

And the output look like this:

| model                                                                        | task            | onnxruntime | tensorrt | torchscript | pplnn | openvino | ncnn |
| :--------------------------------------------------------------------------- | :-------------- | :---------- | :------- | :---------- | :---- | :------- | :--- |
| [DBNet](https://github.com/open-mmlab/mmocr/tree/main/configs/textdet/dbnet) | TextDetection   | Y           | Y        | Y           | Y     | Y        | Y    |
| [CRNN](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/crnn) | TextRecognition | Y           | Y        | Y           | Y     | N        | Y    |
| [SAR](https://github.com/open-mmlab/mmocr/tree/main/configs/textrecog/sar)   | TextRecognition | Y           | N        | N           | N     | N        | N    |
