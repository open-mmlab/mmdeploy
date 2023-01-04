_base_ = ['./classification_static.py']

backend_config = dict(type='ipu', precision='fp16', output_dir='',
                      batches_per_step=1, ipu_version='ipu21', input_shape='input=1,3,224,224', eightbitsio='',
                      popart_options=dict(rearrangeAnchorsOnHost='false', enablePrefetchDatastreams='false', groupHostSync='false', partialsTypeMatMuls="half"))
