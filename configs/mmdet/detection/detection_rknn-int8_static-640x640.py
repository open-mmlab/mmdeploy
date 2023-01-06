_base_ = ['../_base_/base_static.py', '../../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[640, 640])

codebase_config = dict(model_type='rknn')

backend_config = dict(input_size_list=[[3, 640, 640]])

# rtmdet for rknn-toolkit and rknn-toolkit2
# partition_config = dict(
#     type='rknn',  # the partition policy name
#     apply_marks=True,  # should always be set to True
#     partition_cfg=[
#         dict(
#             save_file='model.onnx',  # name to save the partitioned onnx
#             start=['detector_forward:input'],  # [mark_name:input, ...]
#             end=['rtmdet_head:output'],  # [mark_name:output, ...]
#             output_names=[f'pred_maps.{i}' for i in range(6)]) # output names
#     ])
