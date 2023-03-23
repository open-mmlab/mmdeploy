package mmdeploy;

/** @description: the Java API class of TextDetector. */
public class TextDetector {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /** @description: Single text detection result of a picture. */
    public static class Result {

        /** Bbox. */
        public PointF[] bbox;

        /** Score. */
        public float score;

        /** Initializes a new instance of the Result class.
         * @param bbox: bbox.
         * @param score: score.
        */
        public Result(PointF[] bbox, float score) {
            this.bbox = bbox;
            this.score = score;
        }
    }

    /** Initializes a new instance of the TextDetector class.
     * @param modelPath: model path.
     * @param deviceName: device name.
     * @param deviceId: device ID.
     * @exception Exception: create TextDetector failed exception.
    */
    public TextDetector(String modelPath, String deviceName, int deviceId) throws Exception{
        handle = create(modelPath, deviceName, deviceId);
        if (handle == -1) {
            throw new Exception("Create TextDetector failed!");
        }
    }

    /** Get information of each image in a batch.
     * @param images: input mats.
     * @exception Exception: apply TextDetector failed exception.
     * @return: results of each input mat.
    */
    public Result[][] apply(Mat[] images) throws Exception{
        int[] counts = new int[images.length];
        Result[] results = apply(handle, images, counts);
        if (results == null) {
            throw new Exception("Apply TextDetector failed!");
        }
        Result[][] rets = new Result[images.length][];
        int offset = 0;
        for (int i = 0; i < images.length; ++i) {
            Result[] row = new Result[counts[i]];
            if (counts[i] >= 0) {
                System.arraycopy(results, offset, row, 0, counts[i]);
            }
            offset += counts[i];
            rets[i] = row;
        }
        return rets;
    }

    /** Get information of one image.
     * @param image: input mat.
     * @exception Exception: apply TextDetector failed exception.
     * @return: result of input mat.
    */
    public Result[] apply(Mat image) throws Exception{
        int[] counts = new int[1];
        Mat[] images = new Mat[]{image};
        Result[] results = apply(handle, images, counts);
        if (results == null) {
            throw new Exception("Apply TextDetector failed!");
        }
        return results;
    }

    /** Release the instance of TextDetector. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images, int[] count);
}
