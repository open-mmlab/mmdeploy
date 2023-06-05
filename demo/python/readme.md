# MMPose Webcam Demo

![](files/readme.png)

## Get Started

Please make sure `mmdeploy`, `mmpose` (and `mmdetection` which is optional) have been installed on your operating system.

## Usage

Before running the demo, you need to convert the MMPose model via `mmdeploy/tools/deploy.py` first. Please refer to the document if you need help.

```bash
python ${MMDEPLOY_DIR}/demo/python/webcam_demo.py pose_model_path
  [--detect detect_model_path] [--camera camera_id]
	[--device device_name] [--fps] [--skip] [--output] [--code]
```

- `pose_model_path`: The path of MMPose model, which is converted by mmdeploy.
  - Please add the option `--dump-info` when converting model, because the `.json` file are necessary.
- `detect_model_path`: The path of MMDetection model, which is converted by mmdeploy.
  - Because of the same reason, please add the option `--dump-info` when converting model.
- `camera_id`: The id of the camera used as input, or the file name of the input(a video or an image file). `0` by default.
- `device_name`: The device used to run the model. `cuda` by default, or use `cpu` if cuda is not available.
- `fps`: The maximum frames per second. `30` by default.
- `skip`: The model will be run for only 1 in `skip` frames to reduce the pressure. `2` by default.
- `output`: The name of the output file. If not specified, there will be no output file, and the result will be shown in a window.
  - Note that this only works when the input is a video or an image.
- `code`: The coding of the output file if a video is provided as input. `XVID` by default, which corresponding extension is `.avi`.

## FAQ

- `[mmdeploy] [warning] [bulk.h:39] fallback Bulk implementation`
  - It does not matter, you can just ignore it.
