## Usage

1) You should build csharp api first, it will generate a nuget package.

2) Open Demo.sln and install the previous local nuget package. You may refer to [this](https://stackoverflow.com/a/55167481)

3) Generate solution. Select one project as startup and run it.

4) The used model in model can be found from [link1](https://1drv.ms/u/s!Aqis6w3rjKXSh2dXZ5OqbZIZSu9P?e=nefSdY) or [link2](https://pan.baidu.com/s/1VJkLo2oqHos6ZWDT7xamFg?pwd=STAR).
(onnx and tensort(cuda11.1 + cudnn8.2.1 + tensorrt 8.2.3.0 + GTX2070s))
**Note**:
  a) You can convert model on your machine or use the onnx model from the link to test. If you want to use the tensorrt model from the link, make sure your environment and your gpu architecture is same with above.
  b) When you use the downloaded onnx model, you have to edit `deploy.json`, edit `end2end.engine` to `end2end.onnx` and `tensorrt` to `onnxruntime`.
