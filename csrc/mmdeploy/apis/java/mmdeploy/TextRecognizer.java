package mmdeploy;

/**
 * @author: hanrui1sensetime
 * @createDate: 2023/02/28
 * @description: the Java API class of TextRecognizer.
 */
public class TextRecognizer {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    /**
     * @author: hanrui1sensetime
     * @createDate: 2023/02/28
     * @description: Single text recognition result of a picture.
    */
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
    */
    public TextRecognizer(String modelPath, String deviceName, int deviceId) {
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
