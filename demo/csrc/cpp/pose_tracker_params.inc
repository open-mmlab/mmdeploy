// Copyright (c) OpenMMLab. All rights reserved.

DEFINE_int32(det_interval, 1, "Detection interval");
DEFINE_int32(det_label, 0, "Detection label use for pose estimation");
DEFINE_double(det_thr, 0.5, "Detection score threshold");
DEFINE_double(det_min_bbox_size, -1, "Detection minimum bbox size");
DEFINE_double(det_nms_thr, .7,
              "NMS IOU threshold for merging detected bboxes and bboxes from tracked targets");

DEFINE_int32(pose_max_num_bboxes, -1, "Max number of bboxes used for pose estimation per frame");
DEFINE_double(pose_kpt_thr, .5, "Threshold for visible key-points");
DEFINE_int32(pose_min_keypoints, -1,
             "Min number of key-points for valid poses, -1 indicates ceil(n_kpts/2)");
DEFINE_double(pose_bbox_scale, 1.25, "Scale for expanding key-points to bbox");
DEFINE_double(
    pose_min_bbox_size, -1,
    "Min pose bbox size, tracks with bbox size smaller than the threshold will be dropped");
DEFINE_double(pose_nms_thr, 0.5,
              "NMS OKS/IOU threshold for suppressing overlapped poses, useful when multiple pose "
              "estimations collapse to the same target");

DEFINE_double(track_iou_thr, 0.4, "IOU threshold for associating missing tracks");
DEFINE_int32(track_max_missing, 10,
             "Max number of missing frames before a missing tracks is removed");

void InitTrackerParams(mmdeploy::PoseTracker::Params& params) {
  params->det_interval = FLAGS_det_interval;
  params->det_label = FLAGS_det_label;
  params->det_thr = FLAGS_det_thr;
  params->det_min_bbox_size = FLAGS_det_min_bbox_size;
  params->pose_max_num_bboxes = FLAGS_pose_max_num_bboxes;
  params->pose_kpt_thr = FLAGS_pose_kpt_thr;
  params->pose_min_keypoints = FLAGS_pose_min_keypoints;
  params->pose_bbox_scale = FLAGS_pose_bbox_scale;
  params->pose_min_bbox_size = FLAGS_pose_min_bbox_size;
  params->pose_nms_thr = FLAGS_pose_nms_thr;
  params->track_iou_thr = FLAGS_track_iou_thr;
  params->track_max_missing = FLAGS_track_max_missing;
}
