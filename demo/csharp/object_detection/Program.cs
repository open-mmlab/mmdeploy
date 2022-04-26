using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploySharp;

namespace object_detection
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
            Detector handle = new Detector(@"D:\test_model\mask_rcnn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\mask_rcnn\demo.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process
            List<DetectorOutput> output = handle.Apply(mats);

            // 4. show result
            foreach (var obj in output[0].Results)
            {
                if (obj.Score > 0.3)
                {
                    if (obj.HasMask)
                    {
                        Mat imgMask = new Mat(obj.Mask.Height, obj.Mask.Width, MatType.CV_8UC1, obj.Mask.Data);
                        float x0 = Math.Max((float)Math.Floor(obj.BBox.Left) - 1, 0f);
                        float y0 = Math.Max((float)Math.Floor(obj.BBox.Top) - 1, 0f);
                        Rect roi = new Rect((int)x0, (int)y0, obj.Mask.Width, obj.Mask.Height);
                        Cv2.Split(imgs[0], out Mat[] ch);
                        int col = 0;
                        Cv2.BitwiseOr(imgMask, ch[col][roi], ch[col][roi]);
                        Cv2.Merge(ch, imgs[0]);
                    }

                    Cv2.Rectangle(imgs[0], new Point((int)obj.BBox.Left, (int)obj.BBox.Top), new Point((int)obj.BBox.Right, obj.BBox.Bottom), new Scalar(0, 255, 0));
                }
            }
            Cv2.NamedWindow("mmdet", WindowFlags.GuiExpanded);
            Cv2.ImShow("mmdet", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
