package mmdeploy;

public class Segmentor {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /** @description: Single image segmentation result of a picture. */
    public static class Result {

        /** Height. */
        public int height;

        /** Width. */
        public int width;

        /** Number of classes. */
        public int classes;

        /** Segmentation mask. */
        public int[] mask;

        /** Segmentation score. */
        public float[] score;

        /** Initializes a new instance of the Result class.
         * @param height: height.
         * @param width: width.
         * @param classes: number of classes.
         * @param mask: segmentation mask.
         * @param score: segmentation score.
        */
        public Result(int height, int width, int classes, int [] mask, float [] score) {
            this.height = height;
            this.width = width;
            this.classes = classes;
            this.mask = mask;
            this.score = score;
        }
    }

    /** Initializes a new instance of the Segmentor class.
     * @param modelPath: model path.
     * @param deviceName: device name.
     * @param deviceId: device ID.
    */
    public Segmentor(String modelPath, String deviceName, int deviceId) {
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

    /** Release the instance of Segmentor. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images);
}
