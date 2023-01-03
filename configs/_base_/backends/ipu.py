backend_config = dict(type='ipu', precision='fp16', output_dir='/localdata/cn-customer-engineering/qiangg/mmdeploy_popef_hub',
                      batch_per_step=1, ipu_version='ipu21', input_shape='input=1,3,320,320', eightbitsio='',
                      popart_options=dict(rearrangeAnchorsOnHost='false', enablePrefetchDatastreams='false', groupHostSync='false', partialsTypeMatMuls="half"))
