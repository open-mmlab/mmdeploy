backend_config = dict(type='ipu', precision='fp16', output_dir='/localdata/cn-customer-engineering/qiangg/mmdeploy_popef_hub',
                      batch_per_step=1, ipu_version='ipu21', input_shape='input=1,3,224,224', eightbitsio='',
                      popart_options=dict(rearrangeAnchorsOnHost='false', enablePrefetchDatastreams='false', groupHostSync='false', partialsTypeMatMuls="half"))
# backend_config = dict(type='ipu', precision='fp16', output_dir='/localdata/cn-customer-engineering/qiangg/mmdeploy_popef_hub',
#                       batch_per_step=128, ipu_version='ipu21', input_shape='input=88,3,224,224', overlapio='',
#                       popart_options=dict(rearrangeAnchorsOnHost='false', enablePrefetchDatastreams='true', virtualGraphMode=1,
#                        partialsTypeMatMuls="half", numIOTiles=64, enableExplicitMainLoops='true', useHostCopyOps='true', defaultBufferingDepth=2))
