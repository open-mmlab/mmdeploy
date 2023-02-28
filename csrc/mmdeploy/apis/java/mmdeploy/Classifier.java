package mmdeploy;

/**
 * @author: hanrui1sensetime
 * @createDate: 2023/02/28
 * @description: the Java API class of Classifier.
 */
public class Classifier {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /**
     * @author: hanrui1sensetime
     * @createDate: 2023/02/28
     * @description: Single classification result of a picture.
    */
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
    */
    public Classifier(String modelPath, String deviceName, int deviceId) {
        handle = create(modelPath, deviceName, deviceId);
    }

    /** Get label information of each image in a batch.
     * @param images: input mats.
     * @return: results of each input mat.
    */
    public Result[][] apply(Mat[] images) {
        int[] counts = new int[images.length];
        Result[] results = apply(handle, images, counts);
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
    */
    public Result[] apply(Mat image) {
        int[] counts = new int[1];
        Mat[] images = new Mat[]{image};
        return apply(handle, images, counts);
    }

    /** Release the instance of Classifier. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images, int[] count);
}
