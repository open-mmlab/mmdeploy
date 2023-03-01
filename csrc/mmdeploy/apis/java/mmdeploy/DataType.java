package mmdeploy;

/** @description: DataType. */
public enum DataType {
    FLOAT(0),
    HALF(1),
    INT8(2),
    INT32(3);
    final int value;

    /** Initializes a new instance of the DataType class.
     * @param value: the value.
    */
    DataType(int value) {
        this.value = value;
    }
}
