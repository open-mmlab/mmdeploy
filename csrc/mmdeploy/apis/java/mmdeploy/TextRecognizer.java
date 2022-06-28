package mmdeploy;

public class TextRecognizer {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    public static class Result {
        public byte [] text;
        public float [] score;
        public Result(byte [] text, float [] score) {
            this.text = text;
            this.score = score;
        }
    }

    public TextRecognizer(String modelPath, String deviceName, int deviceId) {
        handle = create(modelPath, deviceName, deviceId);
    }

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

    public Result[] apply(Mat image) {
        Mat[] images = new Mat[]{image};
        return apply(handle, images);
    }

    public Result[] applyBbox(Mat image, TextDetector.Result[] bbox, int[] bbox_count) {
        Mat[] images = new Mat[]{image};
        return applyBbox(handle, images, bbox, bbox_count);
    }

    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images);

    private native Result[] applyBbox(long handle, Mat[] images, TextDetector.Result[] bbox, int[] bbox_count);
}
