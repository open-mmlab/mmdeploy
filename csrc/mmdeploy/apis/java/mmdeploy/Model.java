package mmdeploy;

public class Model {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    public final long modelHandle;

    public Model(String path) {
        modelHandle = create(path);
    }

    public void release() {
        destroy(modelHandle);
    }

    private native long create(String path);

    private native void destroy(long modelHandle);
}
