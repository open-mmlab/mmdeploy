package mmdeploy;

/** @description: the Java API class of TextRecognizer. */
public class TextRecognizer {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /** @description: Single text recognition result of a picture. */
    public static class Result {

        /** Text. */
        public byte [] text;

        /** Score. */
        public float [] score;

        /** Initializes a new instance of the Result class.
         * @param text: text.
         * @param score: score.
        */
        public Result(byte [] text, float [] score) {
            this.text = text;
            this.score = score;
        }
    }

    /** Initializes a new instance of the TextRecognizer class.
     * @param modelPath: model path.
     * @param deviceName: device name.
     * @param deviceId: device ID.
     * @exception Exception: create TextRecognizer failed exception.
    */
    public TextRecognizer(String modelPath, String deviceName, int deviceId) throws Exception{
        handle = create(modelPath, deviceName, deviceId);
        if (handle == -1) {
            throw new Exception("Create TextRecognizer failed!");
        }
    }

    /** Get information of each image in a batch.
     * @param images: input mats.
     * @exception Exception: apply TextRecognizer failed exception.
     * @return: results of each input mat.
    */
    public Result[][] apply(Mat[] images) throws Exception{
        Result[] results = apply(handle, images);
        if (results == null) {
            throw new Exception("Apply TextRecognizer failed!");
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
     * @exception Exception: apply TextDetector failed exception.
     * @return: result of input mat.
    */
    public Result[] apply(Mat image) throws Exception{
        Mat[] images = new Mat[]{image};
        Result[] results = apply(handle, images);
        if (results == null) {
            throw new Exception("Apply TextRecognizer failed!");
        }
        return results;
    }

    /** Get information of one image from bboxes.
     * @param image: input mat.
     * @param bbox: bboxes information.
     * @param bbox_count: numter of bboxes
     * @return: result of input mat.
    */
    public Result[] applyBbox(Mat image, TextDetector.Result[] bbox, int[] bbox_count) {
        Mat[] images = new Mat[]{image};
        return applyBbox(handle, images, bbox, bbox_count);
    }

    /** Release the instance of TextRecognizer. */
    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images);

    private native Result[] applyBbox(long handle, Mat[] images, TextDetector.Result[] bbox, int[] bbox_count);
}
