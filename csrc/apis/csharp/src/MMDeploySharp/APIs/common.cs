#pragma warning disable 1591
namespace MMDeploySharp
{
    /// <summary>
    /// Pixel format.
    /// </summary>
    public enum MmPixelFormat
    {
        MM_BGR,
        MM_RGB,
        MM_GRAYSCALE,
        MM_NV12,
        MM_NV21,
        MM_BGRA,
        MM_UNKNOWN_PIXEL_FORMAT
    }

    /// <summary>
    /// Mat data type.
    /// </summary>
    public enum MmDataType
    {
        MM_FLOAT,
        MM_HALF,
        MM_INT8,
        MM_INT32,
        MM_UNKNOWN_DATA_TYPE
    }

    /// <summary>
    /// Function return value.
    /// </summary>
    public enum MmStatus
    {
        MM_SUCCESS = 0,
        MM_E_INVALID_ARG = 1,
        MM_E_NOT_SUPPORTED = 2,
        MM_E_OUT_OF_RANGE = 3,
        MM_E_OUT_OF_MEMORY = 4,
        MM_E_FILE_NOT_EXIST = 5,
        MM_E_FAIL = 6,
        MM_E_UNKNOWN = -1,
    }

    /// <summary>
    /// c struct of mm_mat_t.
    /// </summary>
    public unsafe struct MmMat
    {
        public byte* Data;
        public int Height;
        public int Width;
        public int Channel;
        public MmPixelFormat Format;
        public MmDataType Type;
    }

    /// <summary>
    /// Rect of float value.
    /// </summary>
    public struct MmRect
    {
        public float Left;
        public float Top;
        public float Right;
        public float Bottom;
    }

    /// <summary>
    /// Point of int.
    /// </summary>
    public struct MmPointi
    {
        public int X;
        public int Y;
    }

    /// <summary>
    /// Point of float.
    /// </summary>
    public struct MmPointf
    {
        public float X;
        public float Y;
        public MmPointf(float x, float y)
        {
            X = x;
            Y = y;
        }
    }
}
