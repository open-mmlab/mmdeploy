package mmdeploy;

/** @description: the Java API class of PoseDetector. */
public class PoseDetector {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /** @description: Single pose estimation result of a picture. */
    public static class Result {

        /** Points. */
        public PointF[] point;

        /** Scores of points */
        public float[] score;

        /** Initializes a new instance of the Result class.
         * @param point: points.
         * @param score: scores of points.
        */
        public Result(PointF[] point, float [] score) {
            this.point = point;
            this.score = score;
        }
    }

    /** Initializes a new instance of the PoseDetector class.
     * @param modelPath: model path.
     * @param deviceName: device name.
     * @param deviceId: device ID.
    */
    public PoseDetector(String modelPath, String deviceName, int deviceId) {
        handle = create(modelPath, deviceName, deviceId);
    }

    /** Get information of each image in a batch.
     * @param images: input mats.
     * @return: results of each input mat.
    */
    public Result[][] apply(Mat[] images) {
        Result[] results = apply(handle, images);
        Result[][] rets = new Result[images.length][];
        int offset = 0;
        for (int i = 0; i < images.length; ++i) {
            Result[] row = new Result[1];
            System.arraycopy(results, offset, row, 0, 1);
            offset += 1;
            rets[i] = row;
        }
        return rets;
    }

    /** Get information of one image.
     * @param image: input mat.
     * @return: result of input mat.
    */
    public Result[] apply(Mat image) {
        Mat[] images = new Mat[]{image};
        return apply(handle, images);
    }

    /** Release the instance of PoseDetector. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images);
}
