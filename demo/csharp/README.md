# Usage

**step 0.** install the local nuget package

You should build csharp api first, it will generate a nuget package, or you can download our prebuit package. You may refer to [this](https://stackoverflow.com/a/55167481) on how to install local nuget package.

**step 1.** Add runtime dll to the system path

If you built csharp api from source and didn't build static lib, you should add the built dll to your system path. The same is to opencv, etc.

And don't forget to install backend dependencies. Take tensorrt backend as example, you have to install cudatoolkit, cudnn and tensorrt. The version of backend dependencies that our prebuit nuget package used will be offered in release note.

| backend     | dependencies                  |
| ----------- | ----------------------------- |
| tensorrt    | cudatoolkit, cudnn, tensorrt  |
| onnxruntime | onnxruntime / onnxruntime-gpu |

**step 2.** Open Demo.sln and build solution.

**step 3.** Prepare the model.

You can either convert your model according to this [tutorial](../../docs/en/tutorials/how_to_convert_model.md) or download the test models from [OneDrive](https://1drv.ms/u/s!Aqis6w3rjKXSh2dXZ5OqbZIZSu9P?e=nefSdY) or [BaiduYun](https://pan.baidu.com/s/1VJkLo2oqHos6ZWDT7xamFg?pwd=STAR). The web drive contains onnx and tensorrt models and the test models are converted under environment of cuda11.1 + cudnn8.2.1 + tensorrt 8.2.3.0 + GTX2070s.

*Note*:

- a) If you want to use the tensorrt model from the link, make sure your environment and your gpu architecture is same with above.
- b) When you use the downloaded onnx model, you have to edit `deploy.json`, edit `end2end.engine` to `end2end.onnx` and `tensorrt` to `onnxruntime`.

**step 4.** Set one project as startup project and run it.
