package mmdeploy;

public class Mat {
    public int[] shape;
    public int format;
    public int type;
    public byte[] data;


    public Mat(int height, int width, int channel,
               PixelFormat format, DataType type, byte[] data) {
        shape = new int[]{height, width, channel};
        this.format = format.value;
        this.type = type.value;
        this.data = data;
    }
}
