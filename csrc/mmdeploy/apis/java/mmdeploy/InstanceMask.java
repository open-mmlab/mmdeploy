package mmdeploy;

public class InstanceMask {
    public int[] shape;
    public char[] data;


    public InstanceMask(int height, int width, char[] data) {
        shape = new int[]{height, width};
        this.data = data;
    }
}
