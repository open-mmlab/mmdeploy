_base_ = ['../_base_/base_static.py', '../../_base_/backends/rknn.py']

onnx_config = dict(input_shape=[320, 320])

codebase_config = dict(model_type='rknn')

backend_config = dict(input_size_list=[[3, 320, 320]])

# # yolov3, yolox
# partition_config = dict(
#     type='rknn',  # the partition policy name
#     apply_marks=True,  # should always be set to True
#     partition_cfg=[
#         dict(
#             save_file='model.onnx',  # name to save the partitioned onnx
#             start=['detector_forward:input'],  # [mark_name:input, ...]
#             end=['yolo_head:input'])  # [mark_name:output, ...]
#     ])

# # retinanet, ssd, fsaf
# partition_config = dict(
#     type='rknn',  # the partition policy name
#     apply_marks=True,
#     partition_cfg=[
#         dict(
#             save_file='model.onnx',
#             start='detector_forward:input',
#             end=['BaseDenseHead:output'])
#     ])
