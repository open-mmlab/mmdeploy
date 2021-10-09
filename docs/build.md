## Build MMdeploy

### Build backend support

Update third-party libraries.

```bash
git submodule update --init
```

Build the inference engine extension libraries you need.

- [ONNX Runtime](ops/onnxruntime.md)
- [TensorRT](backends/tensorrt.md)
- [ncnn](backends/ncnn.md)
- [PPL](backends/ppl.md)

### Install mmdeploy

```bash
pip install -e .
```
