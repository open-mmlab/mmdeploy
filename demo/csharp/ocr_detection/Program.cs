using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;

namespace ocr_detection
{
    class Program
    {
        static void CvMatToMat(OpenCvSharp.Mat[] cvMats, out MMDeploy.Mat[] mats)
        {
            mats = new MMDeploy.Mat[cvMats.Length];
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
                Console.WriteLine("usage:\n  ocr_detection deviceName modelPath imagePath\n");
                Environment.Exit(1);
            }

            string deviceName = args[0];
            string modelPath = args[1];
            string imagePath = args[2];

            // 1. create handle
            MMDeploy.TextDetector handle = new MMDeploy.TextDetector(modelPath, deviceName, 0);

            // 2. prepare input
            OpenCvSharp.Mat[] imgs = new OpenCvSharp.Mat[1] { Cv2.ImRead(imagePath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

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
