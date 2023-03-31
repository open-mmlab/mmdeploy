#pragma warning disable 1591
namespace MMDeploy
{
    /// <summary>
    /// Pixel format.
    /// </summary>
    public enum PixelFormat
    {
        BGR,
        RGB,
        Grayscale,
        NV12,
        NV21,
        BGRA,
        UnknownPixelFormat
    }

    /// <summary>
    /// Mat data type.
    /// </summary>
    public enum DataType
    {
        Float,
        Half,
        Int8,
        Int32,
        UnknownDataType
    }

    /// <summary>
    /// Function return value.
    /// </summary>
    public enum Status
    {
        Success = 0,
        InvalidArg = 1,
        NotSupported = 2,
        OutOfRange = 3,
        OutOfMemory = 4,
        FileNotExist = 5,
        Fail = 6,
        Unknown = -1,
    }

    /// <summary>
    /// c struct of mm_mat_t.
    /// </summary>
    public unsafe struct Mat
    {
        public byte* Data;
        public int Height;
        public int Width;
        public int Channel;
        public PixelFormat Format;
        public DataType Type;
        public void* Device;
    }

    /// <summary>
    /// Rect of float value.
    /// </summary>
    public struct Rect
    {
        public float Left;
        public float Top;
        public float Right;
        public float Bottom;
    }

    /// <summary>
    /// Point of int.
    /// </summary>
    public struct Pointi
    {
        public int X;
        public int Y;
    }

    /// <summary>
    /// Point of float.
    /// </summary>
    public struct Pointf
    {
        public float X;
        public float Y;
        public Pointf(float x, float y)
        {
            X = x;
            Y = y;
        }
    }

    /// <summary>
    /// Context type.
    /// </summary>
    public enum ContextType
    {
        DEVICE = 0,
        STREAM = 1,
        MODEL = 2,
        SCHEDULER = 3,
        MAT = 4,
        PROFILER = 5,
    }
}
