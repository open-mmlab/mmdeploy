package mmdeploy;

public class Restorer {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long handle;

    public static class Result {
        public Mat res;
        public Result(Mat res) {
            this.res = res;
        }
    }

    public Restorer(String modelPath, String deviceName, int deviceId) {
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

    public void release() {
        destroy(handle);
    }

    private native long create(String modelPath, String deviceName, int deviceId);

    private native void destroy(long handle);

    private native Result[] apply(long handle, Mat[] images);
}
