using System;
using System.Collections.Generic;
using MMDeploy;
using ImreadModes = OpenCvSharp.ImreadModes;
using Cv2 = OpenCvSharp.Cv2;
using CvMat = OpenCvSharp.Mat;
using MatType = OpenCvSharp.MatType;
using ColorConversionCodes = OpenCvSharp.ColorConversionCodes;
using WindowFlags = OpenCvSharp.WindowFlags;

namespace image_restorer
{
    class Program
    {
        /// <summary>
        /// transform input
        /// </summary>
        static void CvMatToMat(CvMat[] cvMats, out Mat[] mats)
        {
            mats = new Mat[cvMats.Length];
            unsafe
            {
                for (int i = 0; i < cvMats.Length; i++)
                {
                    mats[i].Data = cvMats[i].DataPointer;
                    mats[i].Height = cvMats[i].Height;
                    mats[i].Width = cvMats[i].Width;
                    mats[i].Channel = cvMats[i].Dims;
                    mats[i].Format = PixelFormat.BGR;
                    mats[i].Type = DataType.Int8;
                    mats[i].Device = null;
                }
            }
        }

        static void CvWaitKey()
        {
            Cv2.WaitKey();
        }

        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("usage:\n  image_restorer deviceName modelPath imagePath\n");
                Environment.Exit(1);
            }

            string deviceName = args[0];
            string modelPath = args[1];
            string imagePath = args[2];

            // 1. create handle
            Restorer handle = new Restorer(modelPath, deviceName, 0);

            // 2. prepare input
            CvMat[] imgs = new CvMat[1] { Cv2.ImRead(imagePath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

            // 3. process
            List<RestorerOutput> output = handle.Apply(mats);

            // 4. show result
            CvMat sr_img = new CvMat(output[0].Height, output[0].Width, MatType.CV_8UC3, output[0].Data);
            Cv2.CvtColor(sr_img, sr_img, ColorConversionCodes.RGB2BGR);
            Cv2.NamedWindow("sr", WindowFlags.GuiExpanded);
            Cv2.ImShow("sr", sr_img);
            CvWaitKey();

            handle.Close();
        }
    }
}
