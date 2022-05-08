using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;

namespace pose_detection
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
                }
            }
        }

        static void CvWaitKey()
        {
            Cv2.WaitKey();
        }

        static Vec3b[] GenPalette(int n)
        {
            Random rnd = new Random(2);
            Vec3b[] palette = new Vec3b[n];
            for (int i = 0; i < n; i++)
            {
                byte v1 = (byte)rnd.Next(0, 255);
                byte v2 = (byte)rnd.Next(0, 255);
                byte v3 = (byte)rnd.Next(0, 255);
                palette[i] = new Vec3b(v1, v2, v3);
            }
            return palette;
        }

        static void Main(string[] args)
        {
            if (args.Length != 3)
            {
                Console.WriteLine("usage:\n  pose_detection deviceName modelPath imagePath\n");
                Environment.Exit(1);
            }

            string deviceName = args[0];
            string modelPath = args[1];
            string imagePath = args[2];

            // 1. create handle
            PoseDetector handle = new PoseDetector(modelPath, deviceName, 0);

            // 2. prepare input
            OpenCvSharp.Mat[] imgs = new OpenCvSharp.Mat[1] { Cv2.ImRead(imagePath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

            // 3. process
            List<PoseDetectorOutput> output = handle.Apply(mats);

            // 4. show result
            Vec3b[] palette = GenPalette(output[0].Count);
            int index = 0;
            foreach (var box in output[0].Results)
            {
                for (int i = 0; i < box.Points.Count; i++)
                {
                    Cv2.Circle(imgs[0], (int)box.Points[i].X, (int)box.Points[i].Y, 1,
                        new Scalar(palette[index][0], palette[index][1], palette[index][2]), 2);
                }
                index++;
            }
            Cv2.NamedWindow("pose", WindowFlags.GuiExpanded);
            Cv2.ImShow("pose", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
