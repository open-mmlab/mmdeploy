from mmdeploy.backend.ppl import is_available

__all__ = ['is_available']

if is_available():
    from mmdeploy.backend.ppl import onnx2ppl

    __all__ += ['onnx2ppl']
