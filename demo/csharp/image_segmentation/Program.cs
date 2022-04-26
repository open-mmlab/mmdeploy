using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploySharp;

namespace image_segmentation
{
    class Program
    {
        /// <summary>
        /// transform input
        /// </summary>
        static void CvMatToMmMat(Mat[] cvMats, out MmMat[] mats)
        {
            mats = new MmMat[cvMats.Length];
            unsafe
            {
                for (int i = 0; i < cvMats.Length; i++)
                {
                    mats[i].Data = cvMats[i].DataPointer;
                    mats[i].Height = cvMats[i].Height;
                    mats[i].Width = cvMats[i].Width;
                    mats[i].Channel = cvMats[i].Dims;
                    mats[i].Format = MmPixelFormat.MM_BGR;
                    mats[i].Type = MmDataType.MM_INT8;
                }
            }
        }

        static void CvWaitKey()
        {
            Cv2.WaitKey();
        }

        static void Main(string[] args)
        {
            // 1. create handle
            Segmentor handle = new Segmentor(@"D:\test_model\fcn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\fcn\berlin_000000_000019_leftImg8bit.png", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<SegmentorOutput> output = handle.Apply(mats);

            // 4. show result
            Vec3b[] gen_palette(int classes)
            {
                Random rnd = new Random(0);
                Vec3b[] _palette = new Vec3b[classes];
                for (int i = 0; i < classes; i++)
                {
                    byte v1 = (byte)rnd.Next(0, 255);
                    byte v2 = (byte)rnd.Next(0, 255);
                    byte v3 = (byte)rnd.Next(0, 255);
                    _palette[i] = new Vec3b(v1, v2, v3);
                }
                return _palette;
            }

            Mat color_mask = new Mat(output[0].Height, output[0].Width, MatType.CV_8UC3, new Scalar());
            Vec3b[] palette = gen_palette(output[0].Classes);
            unsafe
            {
                byte* data = color_mask.DataPointer;
                fixed (int* _label = output[0].Mask)
                {
                    int* label = _label;
                    for (int i = 0; i < output[0].Height; i++)
                    {
                        for (int j = 0; j < output[0].Width; j++)
                        {
                            data[0] = palette[*label][0];
                            data[1] = palette[*label][1];
                            data[2] = palette[*label][2];
                            data += 3;
                            label++;
                        }
                    }
                }
            }
            color_mask = imgs[0] * 0.5 + color_mask * 0.5;

            Cv2.NamedWindow("mmseg", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmseg", color_mask);
            CvWaitKey();

            handle.Close();
        }
    }
}
