package mmdeploy;

/** @description: InstanceMask. */
public class InstanceMask {

    /** Mask shape. */
    public int[] shape;

    /** Mask data. */
    public char[] data;

    /** Initialize a new instance of the InstanceMask class.
     * @param height: height.
     * @param width: width.
     * @param data: mask data.
    */
    public InstanceMask(int height, int width, char[] data) {
        shape = new int[]{height, width};
        this.data = data;
    }
}
