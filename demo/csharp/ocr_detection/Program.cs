using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploySharp;

namespace ocr_detection
{
    class Program
    {
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
            MMDeploySharp.TextDetector handle = new MMDeploySharp.TextDetector(@"D:\test_model\dbnet", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\dbnet\demo_text_det.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<TextDetectorOutput> output = handle.Apply(mats);

            // 4. show result
            foreach (var detect in output[0].Results)
            {
                for (int i = 0; i < 4; i++)
                {
                    int sp = i;
                    int ep = (i + 1) % 4;
                    Cv2.Line(imgs[0], new Point((int)detect.BBox[sp].X, (int)detect.BBox[sp].Y),
                        new Point((int)detect.BBox[ep].X, (int)detect.BBox[ep].Y), new Scalar(0, 255, 0));
                }
            }

            Cv2.NamedWindow("ocr-det", WindowFlags.GuiExpanded);
            Cv2.ImShow("ocr-det", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
