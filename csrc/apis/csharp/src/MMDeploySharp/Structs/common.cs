namespace MMDeploySharp
{
    public enum mm_pixel_format_t
    {
        MM_BGR,
        MM_RGB,
        MM_GRAYSCALE,
        MM_NV12,
        MM_NV21,
        MM_BGRA,
        MM_UNKNOWN_PIXEL_FORMAT
    }

    public enum mm_data_type_t
    {
        MM_FLOAT,
        MM_HALF,
        MM_INT8,
        MM_INT32,
        MM_UNKNOWN_DATA_TYPE
    }

    public enum mm_status_t
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

    unsafe public struct mm_mat_t
    {
        public byte* data;
        public int height;
        public int width;
        public int channel;
        public mm_pixel_format_t format;
        public mm_data_type_t type;
    }

    public struct mm_rect_t
    {
        public float left;
        public float top;
        public float right;
        public float bottom;
    }

    public struct mm_pointi_t
    {
        public int x;
        public int y;
    }

    public struct mm_pointf_t
    {
        public float x;
        public float y;
    }
}
