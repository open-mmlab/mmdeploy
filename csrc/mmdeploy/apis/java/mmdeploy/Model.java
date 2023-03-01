package mmdeploy;

/** @description: the Model class. */
public class Model {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long modelHandle;

    /** Initialize a new instance of the Model class.
     * @param path: model path.
    */
    public Model(String path) {
        modelHandle = create(path);
    }

    /** Release the instance of Model. */
    public void release() {
        destroy(modelHandle);
    }

    /** Get model handle.
     * @return: model handle.
    */
    public long handle() {
        return modelHandle;
    }

    private native long create(String path);

    private native void destroy(long modelHandle);
}
