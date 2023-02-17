package mmdeploy;

public class PoseTracker {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;
    private long stateHandle;

    public static class Result {
        public PointF[] keypoints;
        public float[] scores;
        public Rect bbox;
        public int targetID;
        public Result(PointF[] keypoints, float[] scores, Rect bbox, int targetID) {
            this.keypoints = keypoints;
            this.scores = scores;
            this.bbox = bbox;
            this.targetID = targetID;
        }
    }

    public static class Params {
        public int detInterval;
        public int detLabel;
        public float detThr;
        public float detMinBboxSize;
        public float detNmsThr;
        public int poseMaxNumBboxes;
        public float poseKptThr;
        public int poseMinKeypoints;
        public float poseBboxScale;
        public float poseMinBboxSize;
        public float poseNmsThr;
        public float[] keypointSigmas;
        public int keypointSigmasSize;
        public float trackIouThr;
        public int trackMaxMissing;
        public int trackHistorySize;
        public float stdWeightPosition;
        public float stdWeightVelocity;
        public float[] smoothParams;
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
                        this.keypointSigmas = new float[keypointSigmasSize];
                        for (int i = 0; i < keypointSigmasSize; i++) {
                            this.keypointSigmas[i] = keypointSigmas[i];
                        }
                        this.keypointSigmasSize = keypointSigmasSize;
                        this.trackIouThr = trackIouThr;
                        this.trackMaxMissing = trackMaxMissing;
                        this.trackHistorySize = trackHistorySize;
                        this.stdWeightPosition = stdWeightPosition;
                        this.stdWeightVelocity = stdWeightVelocity;
                        this.smoothParams = new float[3];
                        for (int i = 0; i < 3; i++) {
                            this.smoothParams[i] = smoothParams[i];
                        }
                    }
    }

    public PoseTracker(long detect, long pose, long context) {
        handle = create(detect, pose, context);
    }

    public Params initParams() {
        Params params = setDefaultParams();
        return params;
    }

    public long createState(Params params) {
        stateHandle = createState(handle, params);
        return stateHandle;
    }

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

    public Result[] apply(long state, Mat frame, int detect) {
        long[] states = new long[]{state};
        Mat[] frames = new Mat[]{frame};
        int[] detects = new int[]{detect};
        return apply(handle, states, frames, detects);
    }

    public void release() {
        destroy(handle);
    }

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
