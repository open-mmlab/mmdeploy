from mmdeploy.backend.ncnn import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.ncnn.onnx2ncnn import (onnx2ncnn,
                                                 get_output_model_file)
    __all__ += ['onnx2ncnn', 'get_output_model_file']
