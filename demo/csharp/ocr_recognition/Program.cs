using System;
using System.Collections.Generic;
using OpenCvSharp;
using MMDeploy;

namespace ocr_recognition
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
            TextRecognizer handle = new TextRecognizer(@"D:\test_model\crnn", "cuda", 0);

            // 2. prepare input
            Mat[] imgs = new Mat[1] { Cv2.ImRead(@"D:\test_model\crnn\demo_text_recog.jpg", ImreadModes.Color) };
            CvMatToMmMat(imgs, out var mats);

            // 3. process

            List<TextRecognizerOutput> output = handle.Apply(mats);

            //// 4. show result
            foreach (var box in output[0].Results)
            {
                string text = System.Text.Encoding.UTF8.GetString(box.Text);
                Cv2.PutText(imgs[0], text, new Point(20, 20), HersheyFonts.HersheySimplex, 0.7, new Scalar(0, 255, 0), 1);
            }

            Cv2.NamedWindow("ocr-reg", WindowFlags.GuiExpanded);
            Cv2.ImShow("ocr-reg", imgs[0]);
            CvWaitKey();

            handle.Close();
        }
    }
}
