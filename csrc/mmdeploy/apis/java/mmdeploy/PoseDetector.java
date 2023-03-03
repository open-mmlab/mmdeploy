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
     * @exception Exception: create PoseDetector failed exception.
    */
    public PoseDetector(String modelPath, String deviceName, int deviceId) throws Exception{
        handle = create(modelPath, deviceName, deviceId);
        if (handle == -1) {
            throw new Exception("Create PoseDetector failed!");
        }
    }

    /** Get information of each image in a batch.
     * @param images: input mats.
     * @return: results of each input mat.
     * @exception Exception: apply PoseDetector failed exception.
    */
    public Result[][] apply(Mat[] images) throws Exception{
        Result[] results = apply(handle, images);
        if (results == null) {
            throw new Exception("Apply PoseDetector failed!");
        }
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
     * @exception Exception: apply PoseDetector failed exception.
    */
    public Result[] apply(Mat image) throws Exception{
        Mat[] images = new Mat[]{image};
        Result[] results = apply(handle, images);
        if (results == null) {
            throw new Exception("Apply PoseDetector failed!");
        }
        return results;
    }

    /** Release the instance of PoseDetector. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images);
}
