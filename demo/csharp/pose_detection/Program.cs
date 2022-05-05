using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;

namespace pose_detection
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
            PoseDetector handle = new PoseDetector(@"D:\test_model\hrnet", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\hrnet\human-pose.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<PoseDetectorOutput> output = handle.Apply(mats);

            // 4. show result
            Vec3b[] gen_palette(int n)
            {
                Random rnd = new Random(2);
                Vec3b[] _palette = new Vec3b[n];
                for (int i = 0; i < n; i++)
                {
                    byte v1 = (byte)rnd.Next(0, 255);
                    byte v2 = (byte)rnd.Next(0, 255);
                    byte v3 = (byte)rnd.Next(0, 255);
                    _palette[i] = new Vec3b(v1, v2, v3);
                }
                return _palette;
            }
            Vec3b[] palette = gen_palette(output[0].Count);
            int index = 0;
            foreach (var box in output[0].Results)
            {
                for (int i = 0; i < box.Points.Count; i++)
                {
                    Cv2.Circle(imgs[0], (int)box.Points[i].X, (int)box.Points[i].Y, 1, new Scalar(palette[index][0], palette[index][1], palette[index][2]), 2);
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
