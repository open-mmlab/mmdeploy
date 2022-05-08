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
            Detector handle = new Detector(modelPath, deviceName, 0);

            // 2. prepare input
            OpenCvSharp.Mat[] imgs = new OpenCvSharp.Mat[1] { Cv2.ImRead(imagePath, ImreadModes.Color) };
            CvMatToMat(imgs, out var mats);

            // 3. process
            List<DetectorOutput> output = handle.Apply(mats);

            // 4. show result
            foreach (var obj in output[0].Results)
            {
                if (obj.Score > 0.3)
                {
                    if (obj.HasMask)
                    {
                        OpenCvSharp.Mat imgMask = new OpenCvSharp.Mat(obj.Mask.Height, obj.Mask.Width, MatType.CV_8UC1, obj.Mask.Data);
                        float x0 = Math.Max((float)Math.Floor(obj.BBox.Left) - 1, 0f);
                        float y0 = Math.Max((float)Math.Floor(obj.BBox.Top) - 1, 0f);
                        OpenCvSharp.Rect roi = new OpenCvSharp.Rect((int)x0, (int)y0, obj.Mask.Width, obj.Mask.Height);
                        Cv2.Split(imgs[0], out OpenCvSharp.Mat[] ch);
                        int col = 0;
                        Cv2.BitwiseOr(imgMask, ch[col][roi], ch[col][roi]);
                        Cv2.Merge(ch, imgs[0]);
                    }

                    Cv2.Rectangle(imgs[0], new Point((int)obj.BBox.Left, (int)obj.BBox.Top),
                        new Point((int)obj.BBox.Right, obj.BBox.Bottom), new Scalar(0, 255, 0));
                }
            }
            Cv2.NamedWindow("mmdet", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmdet", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
