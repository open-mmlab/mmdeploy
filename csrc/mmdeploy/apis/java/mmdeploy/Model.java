package mmdeploy;

public class Model {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    public final long model_;

    public Model(String path) {
        model_ = create(path);
    }

    public void release() {
        destroy(model_);
    }

    private native long create(String path);

    private native void destroy(long model_);
}
