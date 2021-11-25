from mmdeploy.backend.tensorrt import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.tensorrt.onnx2tensorrt import onnx2tensorrt

    __all__ += ['onnx2tensorrt']
