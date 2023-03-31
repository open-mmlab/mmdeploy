package mmdeploy;

/** @description: the Java API class of Classifier. */
public class Classifier {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /** @description: Single classification result of a picture. */
    public static class Result {

        /** Class id. */
        public int label_id;

        /** Class score. */
        public float score;

        /** Initializes a new instance of the Result class.
         * @param label_id: class id.
         * @param score: class score.
        */
        public Result(int label_id, float score) {
            this.label_id = label_id;
            this.score = score;
        }
    }

    /** Initializes a new instance of the Classifier class.
     * @param modelPath: model path.
     * @param deviceName: device name.
     * @param deviceId: device ID.
     * @exception Exception: create Classifier failed exception.
    */
    public Classifier(String modelPath, String deviceName, int deviceId) throws Exception{
        handle = create(modelPath, deviceName, deviceId);
        if (handle == -1) {
            throw new Exception("Create Classifier failed!");
        }
    }

    /** Get label information of each image in a batch.
     * @param images: input mats.
     * @return: results of each input mat.
     * @exception Exception: apply Classifier failed exception.
    */
    public Result[][] apply(Mat[] images) throws Exception{
        int[] counts = new int[images.length];
        Result[] results = apply(handle, images, counts);
        if (results == null) {
            throw new Exception("Apply Classifier failed!");
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

    /** Get label information of one image.
     * @param image: input mat.
     * @return: result of input mat.
     * @exception Exception: apply Classifier failed exception.
    */
    public Result[] apply(Mat image) throws Exception{
        int[] counts = new int[1];
        Mat[] images = new Mat[]{image};
        Result[] results = apply(handle, images, counts);
        if (results == null) {
            throw new Exception("Apply Classifier failed!");
        }
        return results;
    }

    /** Release the instance of Classifier. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images, int[] count);
}
