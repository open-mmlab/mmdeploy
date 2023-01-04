_base_ = ['../_base_/base_static.py']

backend_config = dict(type='ipu', precision='fp16', output_dir='',
                      batches_per_step=1, ipu_version='ipu21', input_shape='input=1,3,320,320', eightbitsio='',
                      popart_options=dict(rearrangeAnchorsOnHost='false', enablePrefetchDatastreams='false', groupHostSync='false', partialsTypeMatMuls="half"))


onnx_config = dict(input_shape=[800, 1344])
