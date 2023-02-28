package mmdeploy;

public class Model {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long modelHandle;

    public Model(String path) {
        modelHandle = create(path);
    }

    public void release() {
        destroy(modelHandle);
    }

    public long handle() {
        return modelHandle;
    }

    private native long create(String path);

    private native void destroy(long modelHandle);
}
