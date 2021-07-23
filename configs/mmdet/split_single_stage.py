_base_ = ['./base.py']

apply_marks = True

split_params = [
    dict(
        save_file='without_nms.onnx',
        start='detector_forward:input',
        end='multiclass_nms:input')
]
