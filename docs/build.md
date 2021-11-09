## Build MMdeploy

### Build backend support

Update third-party libraries.

```bash
git submodule update --init
```

Install cmake>=3.14.0

Build the inference engine extension libraries you need.

- [ONNX Runtime](backends/onnxruntime.md)
- [TensorRT](backends/tensorrt.md)
- [ncnn](backends/ncnn.md)
- [PPL](backends/ppl.md)
- [OpenVINO](backends/openvino.md)

### Install mmdeploy

```bash
pip install -e .
```
