package mmdeploy;

public class RotatedDetector {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    public static class Result {
        public int label_id;
        public float score;
        public float[] rbbox;
        public Result(int label_id, float score, float[] rbbox) {
            this.label_id = label_id;
            this.score = score;
            this.rbbox = rbbox;
        }
    }

    public RotatedDetector(String modelPath, String deviceName, int deviceId) {
        handle = create(modelPath, deviceName, deviceId);
    }

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

    public Result[] apply(Mat image) {
        int[] counts = new int[1];
        Mat[] images = new Mat[]{image};
        return apply(handle, images, counts);
    }

    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images, int[] count);
}
