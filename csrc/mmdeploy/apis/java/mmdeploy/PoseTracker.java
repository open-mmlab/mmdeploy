package mmdeploy;

/**
 * @author: hanrui1sensetime
 * @createDate: 2023/02/28
 * @description: the Java API class of PoseTracker.
 */
public class PoseTracker {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;
    private long stateHandle;

    /**
     * @author: hanrui1sensetime
     * @createDate: 2023/02/28
     * @description: Single tracking result of a bbox.
    */
    public static class Result {

        /** Keypoints. */
        public PointF[] keypoints;

        /** Scores. */
        public float[] scores;

        /** Bbox */
        public Rect bbox;

        /** Target ID. */
        public int targetID;

        /** Initializes a new instance of the Result class.
         * @param keypoints: keypoints.
         * @param scores: scores.
         * @param bbox: bbox.
         * @param targetID: target ID.
        */
        public Result(PointF[] keypoints, float[] scores, Rect bbox, int targetID) {
            this.keypoints = keypoints;
            this.scores = scores;
            this.bbox = bbox;
            this.targetID = targetID;
        }
    }

    /**
     * @author: hanrui1sensetime
     * @createDate: 2023/02/28
     * @description: PoseTracker parameters.
    */
    public static class Params {

        /** Det interval. */
        public int detInterval;

        /** Det label. */
        public int detLabel;

        /** Det threshold. */
        public float detThr;

        /** Det min bbox size. */
        public float detMinBboxSize;

        /** Det nms threshold. */
        public float detNmsThr;

        /** Pose max number of bboxes. */
        public int poseMaxNumBboxes;

        /** Pose keypoint threshold. */
        public float poseKptThr;

        /** Pose min keypoints. */
        public int poseMinKeypoints;

        /** Pose bbox scale. */
        public float poseBboxScale;

        /** Pose min bbox size. */
        public float poseMinBboxSize;

        /** Pose nms threshold. */
        public float poseNmsThr;

        /** Keypoint sigmas */
        public float[] keypointSigmas;

        /** Keypoint sigmas size. */
        public int keypointSigmasSize;

        /** Track iou threshold. */
        public float trackIouThr;

        /** Track max missing. */
        public int trackMaxMissing;

        /** Track history size. */
        public int trackHistorySize;

        /** std weight position. */
        public float stdWeightPosition;

        /** std weight velocity. */
        public float stdWeightVelocity;

        /** Smooth params. */
        public float[] smoothParams;

        /** Initializes a new instance of the Params class.
         * @param detInterval: det interval.
         * @param detLabel: det label.
         * @param detThr: det threshold.
         * @param detMinBboxSize: det min bbox size.
         * @param detNmsThr: det nms threshold.
         * @param poseMaxNumBboxes: pose max number of bboxes.
         * @param poseKptThr: pose keypoint threshold.
         * @param poseMinKeypoints: pose min keypoints.
         * @param poseBboxScale: pose bbox scale.
         * @param poseMinBboxSize: pose min bbox size.
         * @param poseNmsThr: pose nms threshold.
         * @param keypointSigmas: keypoint sigmas.
         * @param keypointSigmasSize: keypoint sigmas size.
         * @param trackIouThr: track iou threshold.
         * @param trackMaxMissing: track max missing.
         * @param trackHistorySize: track history size.
         * @param stdWeightPosition: std weight position.
         * @param stdWeightVelocity: std weight velocity.
         * @param smoothParams: smooth params.
        */
        public Params(int detInterval, int detLabel, float detThr, float detMinBboxSize, float detNmsThr, int poseMaxNumBboxes,
                    float poseKptThr, int poseMinKeypoints, float poseBboxScale, float poseMinBboxSize, float poseNmsThr, float[] keypointSigmas,
                    int keypointSigmasSize, float trackIouThr, int trackMaxMissing, int trackHistorySize, float stdWeightPosition, float stdWeightVelocity,
                    float[] smoothParams) {
                        this.detInterval = detInterval;
                        this.detLabel = detLabel;
                        this.detThr = detThr;
                        this.detMinBboxSize = detMinBboxSize;
                        this.detNmsThr = detNmsThr;
                        this.poseMaxNumBboxes = poseMaxNumBboxes;
                        this.poseKptThr = poseKptThr;
                        this.poseMinKeypoints = poseMinKeypoints;
                        this.poseBboxScale = poseBboxScale;
                        this.poseMinBboxSize = poseMinBboxSize;
                        this.poseNmsThr = poseNmsThr;
                        this.keypointSigmas = keypointSigmas.clone();
                        this.keypointSigmasSize = keypointSigmasSize;
                        this.trackIouThr = trackIouThr;
                        this.trackMaxMissing = trackMaxMissing;
                        this.trackHistorySize = trackHistorySize;
                        this.stdWeightPosition = stdWeightPosition;
                        this.stdWeightVelocity = stdWeightVelocity;
                        this.smoothParams = smoothParams.clone();
                    }
    }

    /** Initializes a new instance of the PoseTracker class.
     * @param detect: detect model.
     * @param pose: pose model.
     * @param context: context.
    */
    public PoseTracker(Model detect, Model pose, Model context) {
        handle = create(detect.handle(), pose.handle(), context.handle());
    }

    /** Initializes a new instance of the Params class.
     * @return: default value of params.
    */
    public Params initParams() {
        Params params = setDefaultParams();
        return params;
    }

    /** Initializes a new instance of the State class.
     * @param params: params.
     * @return: the handle of State.
    */
    public long createState(Params params) {
        stateHandle = createState(handle, params);
        return stateHandle;
    }

    /** Get information of each frame in a batch.
     * @param states: states of each frame in a batch.
     * @param frames: input mats.
     * @param detects: use detects result or not.
     * @return: results of each input mat.
    */
    public Result[][] apply(long[] states, Mat[] frames, int[] detects) {
        Result[] results = apply(handle, states, frames, detects);
        Result[][] rets = new Result[detects.length][];
        int offset = 0;
        for (int i = 0; i < detects.length; ++i) {
            Result[] row = new Result[1];
            System.arraycopy(results, offset, row, 0, 1);
            offset += 1;
            rets[i] = row;
        }
        return rets;
    }

    /** Get information of one frame.
     * @param state: state of frame.
     * @param frame: input mat.
     * @param detect: use detect result or not.
     * @return: result of input mat.
    */
    public Result[] apply(long state, Mat frame, int detect) {
        long[] states = new long[]{state};
        Mat[] frames = new Mat[]{frame};
        int[] detects = new int[]{detect};
        return apply(handle, states, frames, detects);
    }

    /** Release the instance of PoseTracker. */
    public void release() {
        destroy(handle);
    }

    /** Release the instance of State. */
    public void releaseState(long state) {
        destroyState(state);
    }

    private native long create(long detect, long pose, long context);

    private native void destroy(long handle);

    private native long createState(long pipeline, Params params);

    private native void destroyState(long state);

    public native Params setDefaultParams();

    private native Result[] apply(long handle, long[] states, Mat[] frames, int[] detects);
}
