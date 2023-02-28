package mmdeploy;

public class Device {
    static {
        System.loadLibrary("mmdeploy_java");
    }

    private final long deviceHandle;
    private String deviceName;
    private int deviceIndex;

    public Device(String name, int index) {
        deviceName = name;
        deviceIndex = index;
        deviceHandle = create(deviceName, deviceIndex);
    }

    public String name() {
        return deviceName;
    }

    public int index() {
        return deviceIndex;
    }

    public long handle() {
        return deviceHandle;
    }

    public void release() {
        destroy(deviceHandle);
    }

    private native long create(String name, int index);

    private native void destroy(long deviceHandle);
}
