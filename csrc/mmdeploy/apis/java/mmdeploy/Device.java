package mmdeploy;

public class Device {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    public final long device_;
    private String name_;
    private int index_;

    public Device(String name, int index) {
        name_ = name;
        index_ = index;
        device_ = create(name_, index_);
    }

    public String name() {
        return name_;
    }

    public int index() {
        return index_;
    }

    public void release() {
        destroy(device_);
    }

    private native long create(String name, int index);

    private native void destroy(long device_);
}
