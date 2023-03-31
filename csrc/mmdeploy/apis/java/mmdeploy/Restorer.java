package mmdeploy;

/** @description: the Java API class of Restorer. */
public class Restorer {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /** @description: Single image restore result of a picture. */
    public static class Result {

        /** Result mat. */
        public Mat res;

        /** Initializes a new instance of the Result class.
         * @param res: result mat.
        */
        public Result(Mat res) {
            this.res = res;
        }
    }

    /** Initializes a new instance of the Restorer class.
     * @param modelPath: model path.
     * @param deviceName: device name.
     * @param deviceId: device ID.
     * @exception Exception: create Restorer failed exception.
    */
    public Restorer(String modelPath, String deviceName, int deviceId) throws Exception {
        handle = create(modelPath, deviceName, deviceId);
        if (handle == -1) {
            throw new Exception("Create Restorer failed!");
        }
    }

    /** Get information of each image in a batch.
     * @param images: input mats.
     * @exception Exception: apply Restorer failed exception.
     * @return: results of each input mat.
    */
    public Result[][] apply(Mat[] images) throws Exception {
        Result[] results = apply(handle, images);
        if (results == null) {
            throw new Exception("Apply Restorer failed!");
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
     * @exception Exception: apply Restorer failed exception.
     * @return: result of input mat.
    */
    public Result[] apply(Mat image) throws Exception{
        Mat[] images = new Mat[]{image};
        Result[] results = apply(handle, images);
        if (results == null) {
            throw new Exception("Apply Restorer failed!");
        }
        return results;
    }

    /** Release the instance of Restorer. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images);
}
