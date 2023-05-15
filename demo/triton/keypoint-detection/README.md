python tools/deploy.py configs/mmpose/pose-detection_tensorrt_static-256x192.py ../mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py ../checkpoints/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth demo/resources/human-pose.jpg --work-dir work_dir/hrnet --dump-info --device cuda

python tools/deploy.py configs/mmpose/pose-detection_simcc_tensorrt_dynamic-256x192.py ../mmpose/configs/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192.py https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth demo/resources/human-pose.jpg --work-dir work_dir/pose2 --dump-info --device cuda

