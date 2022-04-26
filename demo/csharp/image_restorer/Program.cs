using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploySharp;

namespace image_restorer
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
            Restorer handle = new Restorer(@"D:\test_model\srcnn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\srcnn\baby.png", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<RestorerOutput> output = handle.Apply(mats);

            // 4. show result
            Mat sr_img = new Mat(output[0].Height, output[0].Width, MatType.CV_8UC3, output[0].Data);
            Cv2.CvtColor(sr_img, sr_img, ColorConversionCodes.RGB2BGR);
            Cv2.NamedWindow("sr", WindowFlags.GuiExpanded);
            Cv2.ImShow("sr", sr_img);
            CvWaitKey();

            handle.Close();
        }
    }
}
