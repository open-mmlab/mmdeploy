using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;

namespace object_detection
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
                Console.WriteLine("usage:\n  object_detection deviceName modelPath imagePath\n");
                Environment.Exit(1);
            }

            string deviceName = args[0];
            string modelPath = args[1];
            string imagePath = args[2];

            // 1. create handle
            RotatedDetector handle = new RotatedDetector(modelPath, deviceName, 0);

            // 2. prepare input
            OpenCvSharp.Mat[] imgs = new OpenCvSharp.Mat[1] { Cv2.ImRead(imagePath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

            // 3. process
            List<RotatedDetectorOutput> output = handle.Apply(mats);

            // 4. show result
            foreach (var obj in output[0].Results)
            {
                if (obj.Score < 0.1)
                {
                    continue;
                }
                float xc = obj.Cx;
                float yc = obj.Cy;
                float wx = obj.Width / 2 * (float)Math.Cos(obj.Angle);
                float wy = obj.Width / 2 * (float)Math.Sin(obj.Angle);
                float hx = -obj.Height / 2 * (float)Math.Sin(obj.Angle);
                float hy = obj.Height / 2 * (float)Math.Cos(obj.Angle);
                OpenCvSharp.Point p1 = new OpenCvSharp.Point(xc - wx - hx, yc - wy - hy);
                OpenCvSharp.Point p2 = new OpenCvSharp.Point(xc + wx - hx, yc + wy - hy);
                OpenCvSharp.Point p3 = new OpenCvSharp.Point(xc + wx + hx, yc + wy + hy);
                OpenCvSharp.Point p4 = new OpenCvSharp.Point(xc - wx + hx, yc - wy + hy);
                var contours = new OpenCvSharp.Point[1][];
                contours[0] = new OpenCvSharp.Point[4] { p1, p2, p3, p4 };
                Cv2.DrawContours(imgs[0], contours, -1, new Scalar(0, 255, 0), 2);
            }
            Cv2.NamedWindow("mmrotate", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmrotate", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
